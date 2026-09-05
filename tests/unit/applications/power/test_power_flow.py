#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.power import (
    Branch,
    Bus,
    BusControl,
    compile_network,
    fixed_mode_power_flow,
    Generator,
    Load,
    PowerBase,
    PowerNetwork,
    PowerStudy,
    Shunt,
    solve_power_flow,
)


def _two_bus(*, p=0.5, q=0.0, q_max=float("inf")):
    network = PowerNetwork(
        (Bus("source", 110), Bus("load", 110)),
        (Branch("line", "source", "load", 0.0, 0.1),),
        (Generator("g", "source", q_max=q_max),),
        (Load("d", "load", p, q),),
    )
    return network, PowerStudy((BusControl("source", "reference"), BusControl("load")))


def test_total_three_phase_bases_and_machine_impedance_rebasing():
    base = PowerBase(100)
    assert base.impedance_ohm(110) == pytest.approx(121)
    assert 3 * base.phase_voltage_volt(110) * base.current_ampere(110) == pytest.approx(
        100e6
    )
    np.testing.assert_allclose(
        base.rebase_impedance(0.2, old_mva=50, old_kv=11, new_kv=110), 0.004
    )


def test_two_bus_analytic_rectangular_sign_loss_and_balance():
    compiled = compile_network(*_two_bus())
    result = solve_power_flow(compiled)
    expected = 0.5 * (1 + np.sqrt(1 - 4 * 0.05**2)) - 0.05j
    assert bool(result.converged), result.status
    np.testing.assert_allclose(result.voltage, [1, expected], atol=2e-7)
    np.testing.assert_allclose(result.branch_to, [-0.5 + 0j], atol=2e-7)
    assert result.branch_from[0].real > 0
    assert result.branch_loss[0].imag > 0
    np.testing.assert_allclose(result.branch_loss.real, 0, atol=2e-7)
    np.testing.assert_allclose(result.total_balance, 0, atol=2e-7)
    np.testing.assert_allclose(result.bus_balance, 0, atol=2e-7)


def test_complex_tap_orientation_and_shunt_inward_power():
    network = PowerNetwork(
        (Bus("h", 110), Bus("l", 11)),
        (Branch("t", "h", "l", 0.01, 0.1, tap=1.1, phase=0.2),),
    )
    result = solve_power_flow(
        network, study=PowerStudy((BusControl("h", "reference"), BusControl("l")))
    )
    assert bool(result.converged)
    np.testing.assert_allclose(result.voltage[1], np.exp(-0.2j) / 1.1, atol=2e-7)
    np.testing.assert_allclose(result.branch_from, 0, atol=2e-7)
    shunt = PowerNetwork((Bus("b", 110),), (), shunts=(Shunt("b", 0.1, 0.2),))
    result = solve_power_flow(shunt, study=PowerStudy((BusControl("b", "reference"),)))
    np.testing.assert_allclose(result.shunt_power, [0.1 - 0.2j], atol=2e-7)
    np.testing.assert_allclose(result.external_reference_power, [0.1 - 0.2j], atol=2e-7)


def test_each_electrical_island_requires_exactly_one_reference():
    disconnected = PowerNetwork((Bus("a"), Bus("b")), ())
    one_reference = PowerStudy((BusControl("a", "reference"), BusControl("b")))
    two_references = PowerStudy(
        (BusControl("a", "reference"), BusControl("b", "reference"))
    )
    with pytest.raises(ValueError, match="exactly one reference"):
        compile_network(disconnected, one_reference)
    connected = PowerNetwork(disconnected.buses, (Branch("ab", "a", "b", 0, 0.1),))
    with pytest.raises(ValueError, match="exactly one reference"):
        compile_network(connected, two_references)
    result = solve_power_flow(disconnected, study=two_references)
    assert bool(result.converged)
    np.testing.assert_allclose(result.voltage, [1, 1], atol=2e-7)


def test_three_bus_pv_saturates_and_preserves_original_balance():
    network = PowerNetwork(
        (Bus("r"), Bus("v"), Bus("d")),
        (Branch("rv", "r", "v", 0, 0.1), Branch("vd", "v", "d", 0, 0.1)),
        (Generator("slack", "r"), Generator("pv", "v", p=0.4, q_min=-0.02, q_max=0.02)),
        (Load("load", "d", 0.8, 0.3),),
    )
    study = PowerStudy(
        (
            BusControl("r", "reference"),
            BusControl("v", "pv", voltage=1.03),
            BusControl("d"),
        )
    )
    result = solve_power_flow(network, study=study)
    assert bool(result.converged), result.status
    assert result.modes == ("reference", "q_max", "pq")
    np.testing.assert_allclose(result.generator_power[1].imag, 0.02, atol=2e-7)
    assert abs(result.voltage[1]) < 1.03
    np.testing.assert_allclose(result.bus_balance, 0, atol=3e-7)
    failed = solve_power_flow(network, study=study, maximum_mode_steps=1)
    assert not bool(failed.converged)
    assert failed.status == "mode_budget_exhausted"


def test_reference_limit_failure_is_not_hidden_as_success_or_pq_conversion():
    network, study = _two_bus(q=0.3, q_max=0.001)
    result = solve_power_flow(network, study=study)
    assert not bool(result.converged)
    assert result.status == "reference_limit_failure"
    assert result.modes[0] == "reference"
    assert result.reference_limit_violation > 0.29
    assert abs(result.bus_balance[0]) > 0.29


def test_fixed_mode_matrix_free_implicit_load_gradient_matches_analytic_voltage():
    compiled = compile_network(*_two_bus())

    def voltage_imaginary(load):
        injections = compiled.specified_power.at[1].set(-load + 0j)
        return fixed_mode_power_flow(compiled, injections).voltage[1].imag

    derivative = jax.grad(voltage_imaginary)(jnp.asarray(0.5))
    np.testing.assert_allclose(derivative, -0.1, atol=2e-6)


def test_one_physical_network_supports_independent_studies_without_inferred_slack():
    network = PowerNetwork((Bus("a"), Bus("b")), (Branch("ab", "a", "b", 0, 0.1),))
    left = PowerStudy((BusControl("a", "reference"), BusControl("b")))
    # Controls may be presented in a different order than the physical buses.
    right = PowerStudy(
        (BusControl("b", "reference", voltage=1.05, angle=0.2), BusControl("a"))
    )
    first = solve_power_flow(network, study=left)
    second = solve_power_flow(network, study=right)
    assert bool(first.converged) and bool(second.converged)
    np.testing.assert_allclose(first.voltage, [1, 1], atol=2e-7)
    np.testing.assert_allclose(second.voltage, [1.05 * np.exp(0.2j)] * 2, atol=2e-7)
    with pytest.raises(ValueError, match="explicit PowerStudy"):
        solve_power_flow(network)
    with pytest.raises(ValueError, match="every physical bus"):
        compile_network(network, PowerStudy((BusControl("a", "reference"),)))
    with pytest.raises(ValueError, match="already bound"):
        solve_power_flow(compile_network(network, left), study=right)
