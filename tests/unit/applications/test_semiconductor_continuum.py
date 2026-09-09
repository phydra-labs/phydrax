#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.semiconductor import (
    DevicePlan,
    DielectricMaterial,
    GateContact,
    mos_capacitor,
    OhmicContact,
    pn_junction,
    PreparedSemiconductorDevice,
    SemiconductorMaterial,
    TransportSupport,
)
from phydrax.discretization import scharfetter_gummel_flux
from phydrax.nonlinear import NonlinearTermination


Q = 1.602176634e-19


@pytest.fixture(autouse=True)
def _double_precision():
    with jax.enable_x64(True):
        yield


def _resistor(*, multidimensional=False):
    if multidimensional:
        support = TransportSupport.tensor_grid(
            (np.linspace(0, 1e-6, 5), np.linspace(0, 0.6e-6, 3)),
            transverse_measure=1e-6,
        )
    else:
        support = TransportSupport.interval(
            np.asarray([0, 0.1, 0.35, 0.8, 1.0]) * 1e-6, area=1e-12
        )
    return PreparedSemiconductorDevice(
        DevicePlan(
            support,
            SemiconductorMaterial.silicon(),
            donor_density=1e20,
            contacts=(
                OhmicContact(
                    "left", support.boundary_patch("left", axis=0, side="lower")
                ),
                OhmicContact(
                    "right", support.boundary_patch("right", axis=0, side="upper")
                ),
            ),
        )
    )


def test_common_fermi_is_exact_detailed_balance_with_full_linear_conductance():
    prepared = PreparedSemiconductorDevice(pn_junction(nodes=9))
    u = prepared.plan.equilibrium_coordinates()
    # Detailed balance must hold for arbitrary potential, not only a converged
    # Poisson solution, including a large internal built-in field.
    u = u.at[:, 0].set(jnp.linspace(-20.0, 20.0, prepared.num_nodes))
    electron, hole = prepared.edge_fluxes(u)
    np.testing.assert_array_equal(electron, 0)
    np.testing.assert_array_equal(hole, 0)
    np.testing.assert_array_equal(prepared.recombination(u), 0)

    direction = jnp.stack(
        (
            jnp.zeros(prepared.num_nodes),
            jnp.linspace(-0.3, 0.5, prepared.num_nodes),
            jnp.linspace(0.2, -0.4, prepared.num_nodes),
        ),
        axis=-1,
    )
    support = prepared.plan.support
    tail, head = support.tail, support.head

    def unfactored(state):
        n, p = prepared.densities(state)
        difference = state[head, 0] - state[tail, 0]
        return (
            support.transmissibility
            * scharfetter_gummel_flux(
                n[tail],
                n[head],
                -difference,
                prepared.plan.thermal_voltage * prepared.plan.electron_mobility[tail],
            ),
            support.transmissibility
            * scharfetter_gummel_flux(
                p[tail],
                p[head],
                difference,
                prepared.plan.thermal_voltage * prepared.plan.hole_mobility[tail],
            ),
        )

    actual = jax.jvp(prepared.edge_fluxes, (u,), (direction,))[1]
    expected = jax.jvp(unfactored, (u,), (direction,))[1]
    # Defends the branch at exactly zero Fermi difference: max/min clipping
    # would silently halve the small-signal conductance there.
    for actual_species, expected_species in zip(actual, expected, strict=True):
        np.testing.assert_allclose(actual_species, expected_species, rtol=2e-13)


def test_srh_is_pair_recombination_and_resolves_near_equilibrium_imbalance():
    prepared = _resistor()
    u = prepared.plan.equilibrium_coordinates()
    u = u.at[:, 1].set(1e-12)
    n, p = prepared.densities(u)
    ni = prepared.plan.intrinsic_density
    expected = (
        ni**2
        * jnp.expm1(1e-12)
        / (
            prepared.plan.hole_lifetime * (n + ni)
            + prepared.plan.electron_lifetime * (p + ni)
        )
    )
    rate = prepared.recombination(u)
    np.testing.assert_allclose(rate, expected, rtol=2e-14)
    assert bool(jnp.all(rate > 0))
    generation = prepared.recombination(u.at[:, 1].set(-1e-12))
    assert bool(jnp.all(generation < 0))


def test_multidimensional_flux_and_srh_satisfy_terminal_charge_continuity():
    prepared = _resistor(multidimensional=True)
    support = prepared.plan.support
    x, y = support.positions[:, 0] / 1e-6, support.positions[:, 1] / 0.6e-6
    u = prepared.plan.equilibrium_coordinates()
    u = u.at[:, 0].add(0.2 * x * y)
    u = u.at[:, 1].set(0.3 * x + 0.1 * y)
    u = u.at[:, 2].set(-0.2 * x * y)
    residual = prepared.residual(u, jnp.zeros(2))
    interior = prepared.plan.semiconductor_mask & ~prepared.plan.ohmic_mask
    integrated_charge_residual = Q * jnp.sum(
        jnp.where(
            interior,
            prepared.count_scale * (residual[:, 1] - residual[:, 2]),
            0,
        )
    )
    currents = prepared.terminal_current(u)
    np.testing.assert_allclose(jnp.sum(currents), integrated_charge_residual, rtol=2e-13)


def test_storage_derivative_is_density_dependent_and_algebraic_rows_are_empty():
    prepared = PreparedSemiconductorDevice(
        mos_capacitor(semiconductor_nodes=5, oxide_nodes=3)
    )
    u = prepared.plan.equilibrium_coordinates()
    direction = jnp.broadcast_to(jnp.asarray([0.3, 0.7, -0.2]), u.shape)
    n, p = prepared.densities(u)
    expected = jnp.stack(
        (
            jnp.zeros(prepared.num_nodes),
            n / prepared.plan.intrinsic_density * (direction[:, 0] + direction[:, 1]),
            p / prepared.plan.intrinsic_density * (-direction[:, 0] - direction[:, 2]),
        ),
        axis=-1,
    )
    expected = jnp.where(prepared.differential_mask, expected, 0)
    derivative = jax.jvp(prepared.storage, (u,), (direction,))[1]
    np.testing.assert_allclose(derivative, expected, rtol=2e-14)
    operator = prepared.storage_jacobian(u)
    np.testing.assert_allclose(
        operator.mv(direction.reshape(-1)).reshape(u.shape), expected, rtol=2e-14
    )
    np.testing.assert_array_equal(prepared.storage(u)[~prepared.differential_mask], 0)


def test_uniform_resistor_has_analytic_positive_into_device_current():
    prepared = _resistor()
    plan = prepared.plan
    voltage = 0.075
    local_voltage = voltage * (1 - plan.support.positions[:, 0] / 1e-6)
    u = plan.equilibrium_coordinates()
    u = u.at[:, 0].add(local_voltage / plan.thermal_voltage)
    u = u.at[:, 1].set(-local_voltage / plan.thermal_voltage)
    u = u.at[:, 2].set(-local_voltage / plan.thermal_voltage)
    n, p = prepared.densities(u)
    conductivity = Q * (plan.electron_mobility[0] * n[0] + plan.hole_mobility[0] * p[0])
    expected = conductivity * 1e-12 * voltage / 1e-6
    np.testing.assert_allclose(
        prepared.terminal_current(u), jnp.asarray([expected, -expected]), rtol=2e-13
    )


def test_pure_dielectric_capacitor_charge_and_displacement_current_signs():
    area, length, permittivity = 1e-12, 1e-6, 3.9 * 8.8541878128e-12
    support = TransportSupport.interval(
        np.asarray([0, 0.1, 0.35, 0.8, 1.0]) * length, area=area
    )
    plan = DevicePlan(
        support,
        DielectricMaterial(
            "oxide",
            permittivity=permittivity,
            provenance="Analytic uniform capacitor regression",
        ),
        contacts=(
            GateContact("left", support.boundary_patch("left", side="lower")),
            GateContact("right", support.boundary_patch("right", side="upper")),
        ),
    )
    prepared = PreparedSemiconductorDevice(plan)
    shape = 1 - support.positions[:, 0] / length
    voltage, slew = 0.4, 2e5
    u = (
        jnp.zeros((prepared.num_nodes, 3))
        .at[:, 0]
        .set(voltage * shape / plan.thermal_voltage)
    )
    udot = jnp.zeros_like(u).at[:, 0].set(slew * shape / plan.thermal_voltage)
    capacitance = permittivity * area / length
    np.testing.assert_allclose(
        prepared.terminal_charge(u),
        np.asarray([1, -1]) * capacitance * voltage,
        rtol=2e-14,
    )
    np.testing.assert_allclose(
        prepared.terminal_current(u, udot),
        np.asarray([1, -1]) * capacitance * slew,
        rtol=2e-14,
    )
    np.testing.assert_array_equal(prepared.terminal_current(u), 0)
    np.testing.assert_array_equal(prepared.storage(u), 0)
    for density in prepared.densities(u):
        np.testing.assert_array_equal(density, 0)
    for flux in prepared.edge_fluxes(u):
        np.testing.assert_array_equal(flux, 0)
    np.testing.assert_allclose(
        prepared.time_scale * prepared.residual(u, jnp.asarray([voltage, 0])),
        0,
        atol=2e-14,
    )


def test_failed_biased_native_solve_never_reports_valid_operating_point():
    prepared = PreparedSemiconductorDevice(pn_junction(nodes=17))
    equilibrium = prepared.equilibrium()
    assert bool(equilibrium.successful)
    np.testing.assert_array_equal(equilibrium.terminal_currents, 0)
    point = prepared.solve(
        jnp.asarray([0.35, 0.0]),
        initial=equilibrium.coordinates,
        termination=NonlinearTermination(
            absolute_residual=1e-12,
            relative_residual=0,
            maximum_steps=1,
        ),
    )
    assert not bool(point.successful)
    assert not bool(point.evidence.valid)
    assert not bool(point.nonlinear_result.successful)
    assert point.evidence.scaled_residual_norm > 1e-12


def test_numeric_refresh_rebinds_material_doping_and_normalization():
    original = _resistor()
    plan = original.plan
    u = plan.equilibrium_coordinates()
    x = plan.support.positions[:, 0] / 1e-6
    u = u.at[:, 0].add(0.2 * x)
    u = u.at[:, 1].set(0.1 * x)
    u = u.at[:, 2].set(-0.15 * x)
    voltage = jnp.asarray([0.01, 0.0])
    native = original.prepare(voltage, u)
    changed_plan = plan.with_doping(3 * plan.donor_density, plan.acceptor_density)
    changed_plan = eqx.tree_at(
        lambda value: (value.electron_mobility, value.hole_mobility, value.permittivity),
        changed_plan,
        (0.7 * plan.electron_mobility, 1.4 * plan.hole_mobility, 0.8 * plan.permittivity),
    )
    changed = eqx.tree_at(lambda value: value.plan, original, changed_plan)
    refreshed = changed.refresh(native, voltage, u)
    direction = jnp.sin(jnp.arange(u.size, dtype=u.dtype) + 0.2)
    expected = jax.jvp(
        lambda flat: changed.residual_function(flat, voltage),
        (u.reshape(-1),),
        (direction,),
    )[1]
    np.testing.assert_allclose(
        refreshed.jacobian.operator.mv(direction), expected, rtol=2e-12, atol=1e-12
    )


def test_failed_equilibrium_prerequisite_survives_a_converged_new_root():
    prepared = PreparedSemiconductorDevice(pn_junction(nodes=17))
    failed = prepared.equilibrium(
        termination=NonlinearTermination(
            absolute_residual=1e-12,
            relative_residual=0,
            maximum_steps=1,
        )
    )
    assert not bool(failed.successful)
    point = prepared.solve(jnp.zeros(2), initial=failed)
    assert bool(point.nonlinear_result.successful)
    assert not bool(point.successful)
    assert not bool(point.evidence.valid)
    assert not bool(point.initial_result.successful)
