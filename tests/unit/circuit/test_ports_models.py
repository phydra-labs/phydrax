import jax.numpy as jnp
import pytest

from phydrax.circuit import (
    audit_scattering,
    ElectricalWaveReference,
    MatrixScatteringComponent,
    power_waves_to_voltage_current,
    rf_common_node_junction,
    ScatteringResponse,
    voltage_current_to_power_waves,
    WavePort,
)


def test_unequal_complex_kurokawa_round_trip_and_power_identity():
    reference = ElectricalWaveReference(jnp.asarray([25.0 + 7.0j, 80.0 - 11.0j]))
    voltage = jnp.asarray([2.0 - 0.5j, -0.2 + 1.4j])
    current = jnp.asarray([0.03 + 0.01j, -0.02 + 0.005j])
    incident, outgoing = voltage_current_to_power_waves(voltage, current, reference)
    recovered_voltage, recovered_current = power_waves_to_voltage_current(
        incident, outgoing, reference
    )
    assert jnp.allclose(recovered_voltage, voltage, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(recovered_current, current, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(
        jnp.abs(incident) ** 2 - jnp.abs(outgoing) ** 2,
        jnp.real(voltage * jnp.conj(current)),
        rtol=1e-12,
        atol=1e-12,
    )


def test_reference_and_response_validation_and_audit():
    with pytest.raises(Exception):
        ElectricalWaveReference(-50.0)
    reference = ElectricalWaveReference(50.0)
    response = ScatteringResponse(
        jnp.diag(jnp.asarray([0.2, 1.2])), (reference, reference), 0
    )
    audit = audit_scattering(response)
    assert not bool(audit.passive)
    assert bool(audit.reciprocal)
    with pytest.raises(ValueError):
        MatrixScatteringComponent(jnp.ones((2, 3)), (WavePort("p", reference),))


def test_unequal_reference_common_node_is_unitary_and_equal_three_port_values():
    junction = rf_common_node_junction((25.0, 50.0, 100.0))
    matrix = junction.evaluate(jnp.asarray(1.0)).matrix
    assert jnp.allclose(jnp.conj(matrix.T) @ matrix, jnp.eye(3), atol=1e-12)
    equal = rf_common_node_junction((50.0, 50.0, 50.0)).evaluate(1.0).matrix
    assert jnp.allclose(jnp.diag(equal), -jnp.ones(3) / 3.0)
    assert jnp.allclose(
        equal - jnp.diag(jnp.diag(equal)), (jnp.ones((3, 3)) - jnp.eye(3)) * 2.0 / 3.0
    )
