import jax.numpy as jnp
import pytest

from phydrax.circuit import (
    Capacitor,
    CircuitInstance,
    ElectricalWaveReference,
    full_mna_scattering_matrix,
    impedance_to_scattering,
    Inductor,
    MatrixScatteringComponent,
    MNASolvePolicy,
    NodalCircuit,
    NodalPort,
    prepare_mna,
    refresh_mna,
    Resistor,
    scattering_to_impedance,
    scattering_to_mna,
    solve_mna,
    WavePort,
)


def test_exp_minus_iwt_component_signs_are_observable():
    omega = jnp.asarray(7.0)
    capacitor = Capacitor(3.0).evaluate(omega)
    assert jnp.allclose(capacitor.y[0, 0], -21.0j)
    inductor = Inductor(5.0).evaluate(omega)
    assert jnp.allclose(inductor.d[0, 0], 35.0j)


def test_grounded_resistor_port_and_floating_rejection():
    reference = ElectricalWaveReference(50.0)
    circuit = NodalCircuit(
        (CircuitInstance("load", Resistor(100.0), ("n", "0")),),
        (NodalPort("p", "n", "0", reference),),
        ground="0",
        circuit_id="resistive-load",
    )
    prepared = prepare_mna(circuit, jnp.asarray(1.0))
    result = solve_mna(prepared, jnp.asarray([[1.0]]))
    assert bool(result.diagnostics.successful)
    assert jnp.allclose(result.outgoing[0, 0], 1.0 / 3.0, atol=1e-12)
    floating = NodalCircuit(
        (CircuitInstance("load", Resistor(100.0), ("a", "b")),),
        (NodalPort("p", "a", "b", reference),),
        ground="0",
        nodes=("0", "a", "b"),
        circuit_id="floating",
    )
    with pytest.raises(ValueError, match="floating"):
        prepare_mna(floating, jnp.asarray(1.0))


def test_complex_reference_s_z_round_trip_and_scattering_mna_parity():
    references = (
        ElectricalWaveReference(25.0 + 4.0j),
        ElectricalWaveReference(80.0 - 3.0j),
    )
    impedance = jnp.asarray([[60.0 + 7.0j, 3.0 - 2.0j], [4.0 + 1.0j, 90.0 - 9.0j]])
    scattering = impedance_to_scattering(impedance, references)
    recovered = scattering_to_impedance(scattering.matrix, references)
    assert bool(jnp.all(scattering.evidence.finite))
    assert jnp.allclose(recovered.matrix, impedance, rtol=1e-10, atol=1e-10)

    reference = ElectricalWaveReference(50.0)
    leaf = MatrixScatteringComponent(
        jnp.asarray([[0.25 + 0.1j]]),
        (WavePort("p", reference),),
    )
    circuit = NodalCircuit(
        (CircuitInstance("s", scattering_to_mna(leaf), ("n",)),),
        (NodalPort("p", "n", "0", reference),),
        ground="0",
        circuit_id="s-lowered",
    )
    matrix = full_mna_scattering_matrix(prepare_mna(circuit, jnp.asarray(2.0)))
    assert jnp.allclose(matrix, leaf.evaluate(jnp.asarray(2.0)).matrix, atol=1e-10)


def test_sparse_mna_assembles_and_solves_without_dense_materialization():
    reference = ElectricalWaveReference(50.0)
    circuit = NodalCircuit(
        (CircuitInstance("load", Resistor(100.0), ("n", "0")),),
        (NodalPort("p", "n", "0", reference),),
        ground="0",
        circuit_id="sparse-resistive-load",
    )
    prepared = prepare_mna(
        circuit,
        jnp.asarray(1.0),
        MNASolvePolicy(assembly="sparse", residual_tolerance=1e-8),
    )
    assert prepared.matrix is None
    assert prepared.sparse_operator is not None
    assert prepared.plan.linear_plan.backend == "native-krylov"
    result = solve_mna(prepared, jnp.asarray([[1.0]]))
    assert bool(result.diagnostics.successful)
    assert jnp.allclose(result.outgoing[0, 0], 1.0 / 3.0, atol=1e-8)
    refreshed_circuit = NodalCircuit(
        (CircuitInstance("load", Resistor(200.0), ("n", "0")),),
        (NodalPort("p", "n", "0", reference),),
        ground="0",
        circuit_id="sparse-resistive-load",
    )
    refreshed = refresh_mna(prepared, refreshed_circuit, jnp.asarray(1.0))
    assert refreshed.matrix is None and refreshed.sparse_operator is not None
    refreshed_result = solve_mna(refreshed, jnp.asarray([[1.0]]))
    assert jnp.allclose(refreshed_result.outgoing[0, 0], 0.6, atol=1e-8)

    two_port = NodalCircuit(
        (CircuitInstance("series", Resistor(75.0), ("left", "right")),),
        (
            NodalPort("left", "left", "0", reference),
            NodalPort("right", "right", "0", reference),
        ),
        ground="0",
        circuit_id="sparse-two-port",
    )
    dense_matrix = full_mna_scattering_matrix(prepare_mna(two_port, jnp.asarray(2.0)))
    sparse_matrix = full_mna_scattering_matrix(
        prepare_mna(
            two_port,
            jnp.asarray(2.0),
            MNASolvePolicy(assembly="sparse", residual_tolerance=1e-8),
        )
    )
    assert jnp.allclose(sparse_matrix, dense_matrix, rtol=1e-7, atol=1e-8)
