import equinox as eqx
import jax.numpy as jnp

import phydrax as phx


class _TableAmplitude(eqx.Module):
    log_abs: jnp.ndarray
    phase: jnp.ndarray

    def __call__(self, configuration):
        bits = (configuration > 0).astype(jnp.int32)
        index = 2 * bits[0] + bits[1]
        return phx.operators.LogAmplitude(self.log_abs[index], self.phase[index])


def _ising_operator(coupling=0.7, field=0.4):
    def diagonal(configurations):
        return -coupling * configurations[..., 0] * configurations[..., 1]

    def connections(configurations):
        first = configurations.at[..., 0].multiply(-1)
        second = configurations.at[..., 1].multiply(-1)
        connected = jnp.stack((first, second), axis=-2)
        shape = configurations.shape[:-1] + (2,)
        return phx.operators.ConnectedConfigurations(
            connected,
            -field * jnp.ones(shape),
            jnp.ones(shape, dtype=bool),
            configuration_shape=(2,),
        )

    return phx.operators.CallableDiscreteQuantumOperator(
        diagonal,
        connections,
        configuration_shape=(2,),
        operator_id="two-spin-ising",
    )


def _configurations():
    return jnp.asarray([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=jnp.int32)


def test_connected_local_estimator_matches_dense_complex_hamiltonian():
    configurations = _configurations()
    amplitudes = jnp.asarray([1.0 + 0.0j, 0.7 + 0.2j, -0.3 + 0.9j, 1.2 - 0.4j])
    model = _TableAmplitude(
        jnp.log(jnp.abs(amplitudes)), amplitudes / jnp.abs(amplitudes)
    )
    operator = _ising_operator()
    local = phx.operators.evaluate_local_operator(model, operator, configurations)

    coupling = 0.7
    field = 0.4
    dense = jnp.zeros((4, 4), dtype=complex)
    for index, configuration in enumerate(configurations):
        dense = dense.at[index, index].set(
            -coupling * configuration[0] * configuration[1]
        )
        dense = dense.at[index, index ^ 2].set(-field)
        dense = dense.at[index, index ^ 1].set(-field)
    expected = dense @ amplitudes / amplitudes

    assert isinstance(local, phx.operators.LocalOperatorEstimate)
    assert isinstance(operator, phx.operators.AbstractLocalQuantumOperator)
    assert local.configuration_shape == operator.configuration_shape
    assert local.operator_id == operator.operator_id
    assert local.method_id == "connected-configurations"
    assert jnp.all(
        local.status == int(phx.operators.LocalOperatorStatus.SUCCESS)
    )
    assert jnp.all(local.valid)
    assert jnp.array_equal(local.work_count, jnp.full((4,), 2))
    assert jnp.allclose(local.value, expected)


def test_padded_connections_do_not_change_local_estimate():
    configurations = _configurations()
    model = _TableAmplitude(jnp.zeros((4,)), jnp.ones((4,), dtype=complex))

    def diagonal(values):
        return jnp.zeros(values.shape[:-1])

    def connections(values):
        first = values.at[..., 0].multiply(-1)
        padding = jnp.zeros_like(values)
        connected = jnp.stack((first, padding), axis=-2)
        shape = values.shape[:-1] + (2,)
        return phx.operators.ConnectedConfigurations(
            connected,
            jnp.asarray([1.0, jnp.nan]) * jnp.ones(shape),
            jnp.asarray([True, False]) * jnp.ones(shape, dtype=bool),
            configuration_shape=(2,),
        )

    operator = phx.operators.CallableDiscreteQuantumOperator(
        diagonal,
        connections,
        configuration_shape=(2,),
        operator_id="padded",
    )
    local = phx.operators.evaluate_local_operator(model, operator, configurations)

    assert jnp.all(local.valid)
    assert jnp.allclose(local.value, 1.0)
    assert jnp.array_equal(local.work_count, jnp.ones((4,), dtype=jnp.int32))
    gradient = eqx.filter_grad(
        lambda amplitude: jnp.real(
            jnp.sum(
                phx.operators.evaluate_local_operator(
                    amplitude, operator, configurations
                ).value
            )
        )
    )(model)
    assert jnp.all(jnp.isfinite(gradient.log_abs))
    assert jnp.all(jnp.isfinite(gradient.phase))


def test_zero_current_amplitude_invalidates_local_estimator():
    configurations = _configurations()
    model = _TableAmplitude(
        jnp.asarray([-jnp.inf, 0.0, 0.0, 0.0]),
        jnp.ones((4,), dtype=complex),
    )
    local = phx.operators.evaluate_local_operator(
        model, _ising_operator(), configurations
    )

    assert not local.valid[0]
    assert jnp.isnan(local.value[0])
    assert jnp.all(local.valid[1:])
