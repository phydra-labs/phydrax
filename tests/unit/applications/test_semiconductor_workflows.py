#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Independent continuum-device limits, not assertions about solver plumbing."""

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.applications import semiconductor as sc


jax.config.update("jax_enable_x64", True)
Q = 1.602176634e-19


def test_abrupt_pn_equilibrium_resolves_depletion_field_and_mesh_converges():
    peaks = []
    for nodes in (41, 81):
        plan = sc.pn_junction(nodes)
        device = sc.PreparedSemiconductorDevice(plan)
        point = device.equilibrium()
        assert bool(point.successful)
        n, p = device.densities(point.coordinates)
        np.testing.assert_allclose(n * p, plan.intrinsic_density**2, rtol=2e-12)
        psi = np.asarray(point.coordinates[:, 0]) * float(plan.thermal_voltage)
        x = np.asarray(plan.support.positions[:, 0])
        peaks.append(float(np.max(np.abs(np.diff(psi) / np.diff(x)))))
        # Depletion approximation: equal doping gives triangular electric field.
        vt = float(plan.thermal_voltage)
        ni = float(plan.intrinsic_density[0])
        doping = 1e21
        builtin = 2 * vt * np.log(doping / ni)
        width = np.sqrt(4 * float(plan.permittivity[0]) * builtin / (Q * doping))
        reference_peak = 2 * builtin / width
        np.testing.assert_allclose(psi[-1] - psi[0], builtin, rtol=2e-6)
        # Thermal transition tails are absent from the depletion approximation.
        np.testing.assert_allclose(peaks[-1], reference_peak, rtol=0.16)
        assert np.max(np.abs(np.asarray(point.terminal_currents))) < 1e-15
    np.testing.assert_allclose(peaks[1], peaks[0], rtol=0.06)


def test_flatband_mos_capacitance_matches_oxide_and_debye_series_limit():
    provisional = sc.mos_capacitor(81, 11)
    offset = float(provisional.contact_potential_offset[-1])
    plan = sc.mos_capacitor(81, 11, gate_offset=offset)
    device = sc.PreparedSemiconductorDevice(plan)
    equilibrium = device.equilibrium()
    assert bool(equilibrium.successful)
    delta = 2e-4
    minus = device.solve(jnp.asarray([-delta, 0.0]), initial=equilibrium.coordinates)
    plus = device.solve(jnp.asarray([delta, 0.0]), initial=equilibrium.coordinates)
    assert bool(minus.successful & plus.successful)
    measured = float((plus.terminal_charges[0] - minus.terminal_charges[0]) / (2 * delta))
    n, p = device.densities(equilibrium.coordinates)
    eps_si = float(plan.permittivity[-1])
    screening = np.sqrt(eps_si * float(plan.thermal_voltage) / (Q * float(n[-1] + p[-1])))
    area = 1e-12
    oxide_capacitance = float(plan.permittivity[0]) * area / 50e-9
    debye_capacitance = eps_si * area / screening
    expected = 1 / (1 / oxide_capacitance + 1 / debye_capacitance)
    assert measured > 0
    np.testing.assert_allclose(measured, expected, rtol=0.08)
    # The gate is an electrostatic terminal, not an Ohmic carrier source.
    assert abs(float(plus.terminal_currents[0])) < 1e-18


def test_lateral_bjt_base_bias_controls_collector_current_with_terminal_kcl():
    plan = sc.bipolar_transistor(21, 5)
    device = sc.PreparedSemiconductorDevice(plan)
    equilibrium = device.equilibrium()
    assert bool(equilibrium.successful)
    path = jnp.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.025, 0.1],
            [0.0, 0.05, 0.1],
            [0.0, 0.025, 0.1],
        ]
    )
    sweep = device.sweep(path, initial=equilibrium.coordinates)
    assert bool(sweep.successful)
    np.testing.assert_array_equal(sweep.voltages, path)
    lower, upper, returned = sweep.points[1:]
    np.testing.assert_allclose(
        returned.terminal_currents,
        lower.terminal_currents,
        rtol=1e-5,
        atol=1e-18,
    )
    for point in (lower, upper):
        currents = np.asarray(point.terminal_currents)
        assert currents[2] > 0  # positive conventional current enters collector
        assert currents[0] < 0  # leaves emitter
        assert abs(currents.sum()) < 1e-7 * np.max(np.abs(currents)) + 1e-17
    assert float(upper.terminal_currents[2]) > float(lower.terminal_currents[2])
