import jax
import jax.numpy as jnp

from phydrax.circuit import (
    ElectricalWaveReference,
    MatrixScatteringComponent,
    read_touchstone,
    SampledScatteringModel,
    scattering_least_squares_problem,
    ScatteringDataset,
    TouchstoneData,
    WavePort,
    write_touchstone,
)


def test_touchstone_column_order_references_round_trip_and_exact_nodes(tmp_path):
    frequencies = jnp.asarray([1.0e9, 2.0e9])
    matrices = jnp.asarray(
        [
            [[1.0 + 1.0j, 2.0 + 2.0j], [3.0 + 3.0j, 4.0 + 4.0j]],
            [[5.0 + 1.0j, 6.0 + 2.0j], [7.0 + 3.0j, 8.0 + 4.0j]],
        ]
    )
    data = TouchstoneData(
        frequencies,
        matrices,
        jnp.asarray([40.0, 75.0]),
        port_names=("input", "output"),
        data_format="RI",
        frequency_unit="GHZ",
        version="2.0",
    )
    path = tmp_path / "asymmetric.s2p"
    write_touchstone(path, data)
    parsed = read_touchstone(path)
    assert parsed.port_names == data.port_names
    assert jnp.array_equal(parsed.reference_impedance, data.reference_impedance)
    assert jnp.allclose(parsed.scattering, matrices)
    model = SampledScatteringModel.from_touchstone(parsed)
    evaluated = model.evaluate(2.0 * jnp.pi * frequencies)
    assert jnp.allclose(evaluated.matrix, matrices)


def test_calibration_adapter_preserves_native_gradients():
    reference = ElectricalWaveReference(50.0)
    ports = (WavePort("p", reference),)
    target = jnp.asarray([[[0.3 + 0.0j]]])
    dataset = ScatteringDataset(
        jnp.asarray([1.0]),
        target,
        (reference,),
        port_ids=("p",),
    )

    def parameterize(theta):
        return MatrixScatteringComponent(jnp.asarray([[jnp.tanh(theta)]]), ports)

    problem = scattering_least_squares_problem(parameterize, (dataset,))

    def objective(theta):
        residual, _ = problem.value(theta)
        return 0.5 * jnp.sum(residual**2)

    theta = jnp.asarray(0.2)
    gradient = jax.grad(objective)(theta)
    step = 1e-5
    finite_difference = (objective(theta + step) - objective(theta - step)) / (2.0 * step)
    assert jnp.allclose(gradient, finite_difference, rtol=2e-4, atol=2e-5)
