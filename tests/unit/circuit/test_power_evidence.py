import equinox as eqx
import jax.numpy as jnp

from phydrax.circuit import (
    AdmittanceComponent,
    assess_circuit_energy_ledger,
    assess_mna_power_ledger,
    Capacitor,
    CircuitElement,
    CircuitInstance,
    ElectricalWaveReference,
    evaluate_circuit_energy_ledger,
    evaluate_harmonic_balance_energy_ledger,
    evaluate_mna_power_ledger,
    ExponentialDiodeLaw,
    IndependentCurrentSourceLaw,
    Inductor,
    NodalCircuit,
    NodalPort,
    prepare_circuit_dae,
    prepare_mna,
    Resistor,
    solve_harmonic_balance,
    solve_mna,
    TemporalHarmonicPlan,
)


def _reference():
    return ElectricalWaveReference(50.0)


def _parallel_rlc_with_current_source():
    source = CircuitElement(
        IndependentCurrentSourceLaw(1.0, input_key="drive"),
        element_id="source",
    )
    return NodalCircuit(
        (
            CircuitInstance("resistor", Resistor(2.0), ("n", "0")),
            CircuitInstance("capacitor", Capacitor(1.0), ("n", "0")),
            CircuitInstance("inductor", Inductor(1.0), ("n", "0")),
            CircuitInstance("source", source, ("0", "n")),
        ),
        (NodalPort("terminal", "n", "0", _reference()),),
        ground="0",
        circuit_id="parallel-rlc-driven",
    )


def test_mna_rlc_phasor_ledger_and_corrupted_contribution():
    circuit = NodalCircuit(
        (
            CircuitInstance("resistor", Resistor(2.0), ("n", "0")),
            CircuitInstance("capacitor", Capacitor(1.0), ("n", "0")),
            CircuitInstance("inductor", Inductor(1.0), ("n", "0")),
        ),
        (NodalPort("terminal", "n", "0", _reference()),),
        ground="0",
        circuit_id="parallel-rlc-phasor",
    )
    prepared = prepare_mna(circuit, jnp.asarray(1.0))
    solved = solve_mna(prepared, jnp.asarray([[1.0 + 0.25j]]))
    ledger = evaluate_mna_power_ledger(prepared, solved, closure_tolerance=1e-10)

    assert ledger.phasor_amplitude_convention == "rms"
    assert ledger.current_orientation == "ports-into-circuit;elements-into-device"
    assert ledger.element_ids == ("resistor", "capacitor", "inductor")
    assert ledger.source_ids == ()
    assert bool(ledger.available)
    assert bool(jnp.all(ledger.real_power_closed))
    assert bool(jnp.all(ledger.reactive_power_closed))
    assert ledger.element_reactive_power[1, 0] > 0.0
    assert ledger.element_reactive_power[2, 0] < 0.0

    corrupted = eqx.tree_at(
        lambda value: value.element_real_power,
        ledger,
        ledger.element_real_power.at[0, 0].add(0.25),
    )
    reassessed = assess_mna_power_ledger(corrupted)
    assert not bool(jnp.all(reassessed.real_power_closed))
    assert jnp.abs(reassessed.real_power_residual[0]) > 0.2


def test_mna_black_box_power_evidence_is_explicitly_unavailable():
    black_box = AdmittanceComponent(
        jnp.asarray([[1.0, -1.0], [-1.0, 1.0]]),
        component_id="black-box",
    )
    circuit = NodalCircuit(
        (CircuitInstance("black-box", black_box, ("n", "0")),),
        (NodalPort("terminal", "n", "0", _reference()),),
        ground="0",
        circuit_id="unsupported-phasor-power",
    )
    prepared = prepare_mna(circuit, jnp.asarray(2.0))
    ledger = evaluate_mna_power_ledger(
        prepared, solve_mna(prepared, jnp.asarray([[1.0]]))
    )

    assert not bool(ledger.available)
    assert ledger.unsupported_element_ids == ("black-box",)
    assert "no supported element power law" in ledger.unavailable_reasons[0]
    assert not bool(jnp.any(ledger.real_power_closed))
    assert jnp.all(jnp.isnan(ledger.real_power_residual))


def test_transient_rlc_energy_ledger_closes_with_separate_source_power():
    prepared = prepare_circuit_dae(_parallel_rlc_with_current_source())
    times = jnp.linspace(0.0, 1.0, 257)
    voltage = jnp.sin(times)
    inductor_current = -jnp.cos(times)
    states = jnp.stack((voltage, inductor_current), axis=-1)
    rates = jnp.stack((jnp.cos(times), jnp.sin(times)), axis=-1)
    args = {"inputs": {"drive": lambda time: 0.5 * jnp.sin(time)}}
    ledger = evaluate_circuit_energy_ledger(
        prepared,
        times,
        states,
        rates,
        args=args,
        port_currents=jnp.zeros((times.size, 1)),
        closure_tolerance=2e-5,
    )

    assert ledger.element_ids == ("resistor", "capacitor", "inductor")
    assert ledger.source_ids == ("source",)
    assert bool(ledger.available)
    assert bool(ledger.passive_dissipation_valid)
    assert bool(ledger.closed)
    assert jnp.allclose(ledger.balance_residual, 0.0, atol=1e-12)
    assert jnp.all(ledger.element_dissipated_power[:, 0] >= 0.0)
    assert jnp.allclose(
        ledger.element_dissipated_power[:, 0],
        -ledger.source_power[:, 0],
        atol=1e-12,
    )
    corrupted = eqx.tree_at(
        lambda value: value.element_dissipated_power,
        ledger,
        ledger.element_dissipated_power.at[100, 0].add(0.1),
    )
    assert not bool(assess_circuit_energy_ledger(corrupted).closed)


def test_source_power_sign_reverses_without_becoming_dissipation():
    source = CircuitElement(
        IndependentCurrentSourceLaw(1.0, input_key="drive"),
        element_id="source",
    )
    circuit = NodalCircuit(
        (
            CircuitInstance("capacitor", Capacitor(1.0), ("n", "0")),
            CircuitInstance("source", source, ("0", "n")),
        ),
        (NodalPort("terminal", "n", "0", _reference()),),
        ground="0",
        circuit_id="source-sign",
    )
    prepared = prepare_circuit_dae(circuit)
    times = jnp.asarray([0.0, 1.0])
    supplied = evaluate_circuit_energy_ledger(
        prepared,
        times,
        jnp.asarray([[1.0], [2.0]]),
        jnp.ones((2, 1)),
        args={"inputs": {"drive": 1.0}},
        port_currents=jnp.zeros((2, 1)),
        closure_tolerance=1e-12,
    )
    absorbed = evaluate_circuit_energy_ledger(
        prepared,
        times,
        jnp.asarray([[2.0], [1.0]]),
        -jnp.ones((2, 1)),
        args={"inputs": {"drive": -1.0}},
        port_currents=jnp.zeros((2, 1)),
        closure_tolerance=1e-12,
    )

    assert bool(supplied.closed) and bool(absorbed.closed)
    assert jnp.all(supplied.source_power[:, 0] < 0.0)
    assert jnp.all(absorbed.source_power[:, 0] > 0.0)
    assert jnp.allclose(supplied.element_dissipated_power, 0.0)
    assert jnp.allclose(absorbed.element_dissipated_power, 0.0)
    assert bool(supplied.passive_dissipation_valid)
    assert bool(absorbed.passive_dissipation_valid)


def test_driven_periodic_rlc_energy_ledger_integrates_one_period():
    prepared = prepare_circuit_dae(_parallel_rlc_with_current_source())
    temporal = TemporalHarmonicPlan(jnp.asarray(1.0), 17, prepared.plan.layout.size)
    times = temporal.times
    initial = jnp.stack((jnp.sin(times), -jnp.cos(times)), axis=-1)
    args = {"inputs": {"drive": lambda time: 0.5 * jnp.sin(time)}}
    solved = solve_harmonic_balance(
        prepared,
        initial,
        jnp.asarray(1.0),
        args=args,
    )
    ledger = evaluate_harmonic_balance_energy_ledger(
        prepared,
        solved,
        args=args,
        port_currents=jnp.zeros((times.size, 1)),
        closure_tolerance=1e-8,
    )

    assert bool(ledger.available)
    assert bool(ledger.aliasing_tail_valid)
    assert bool(ledger.closed)
    assert jnp.abs(ledger.endpoint_energy_defect) < 1e-10
    assert jnp.abs(ledger.period_balance_defect) < 1e-10
    assert jnp.allclose(
        ledger.element_dissipated_energy[0],
        -ledger.source_energy[0],
        atol=1e-10,
    )


def test_transient_element_without_energy_law_is_unavailable():
    diode = CircuitElement(
        ExponentialDiodeLaw(1e-12, 0.025),
        element_id="diode",
    )
    circuit = NodalCircuit(
        (CircuitInstance("diode", diode, ("n", "0")),),
        (NodalPort("terminal", "n", "0", _reference()),),
        ground="0",
        circuit_id="unsupported-transient-energy",
    )
    prepared = prepare_circuit_dae(circuit)
    ledger = evaluate_circuit_energy_ledger(
        prepared,
        jnp.asarray([0.0, 1.0]),
        jnp.zeros((2, 1)),
        jnp.zeros((2, 1)),
        port_currents=jnp.zeros((2, 1)),
    )

    assert not bool(ledger.available)
    assert bool(ledger.finite)
    assert not bool(ledger.closed)
    assert ledger.unsupported_element_ids == ("diode",)
    assert "no passive energy law" in ledger.unavailable_reasons[0]
