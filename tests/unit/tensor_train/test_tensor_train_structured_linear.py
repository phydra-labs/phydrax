import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax.tensor_train as tt


def test_structured_identity_shift_boundaries_and_dense_budgets():
    vector = jnp.arange(4, dtype=jnp.float32)
    train = tt.TensorTrain((vector[None, :, None],))
    identity = tt.cartesian_identity((4,))
    periodic = tt.shift_operator(
        (4,), 0, offset=1, boundary=tt.BoundaryPolicy("periodic")
    )
    dirichlet = tt.shift_operator(
        (4,), 0, offset=1, boundary=tt.BoundaryPolicy("dirichlet")
    )
    neumann = tt.shift_operator((4,), 0, offset=1, boundary=tt.BoundaryPolicy("neumann"))

    assert jnp.array_equal(identity.apply(train).to_dense(max_entries=4), vector)
    assert jnp.array_equal(
        periodic.apply(train).to_dense(max_entries=4), jnp.roll(vector, 1)
    )
    assert jnp.array_equal(
        dirichlet.apply(train).to_dense(max_entries=4), jnp.asarray([0.0, 0.0, 1.0, 2.0])
    )
    assert jnp.array_equal(
        neumann.apply(train).to_dense(max_entries=4), jnp.asarray([0.0, 0.0, 1.0, 2.0])
    )
    with pytest.raises(ValueError, match="exceeding budget"):
        identity.to_matrix(max_entries=15)
    with pytest.raises(ValueError, match="exceeding budget"):
        train.to_dense(max_entries=3)


def test_qtt_sampling_returns_physical_indices_without_quantum_semantics():
    layout = tt.QuanticsLayout.binary((4,), ordering="blocked")
    mass = jnp.asarray([1.0, 2.0, 3.0, 4.0])
    train = tt.qtt_digitize(mass, layout, max_ranks=2, relative_tolerance=0.0).tensor
    samples = tt.qtt_sample(
        train,
        layout,
        jax.random.key(8),
        sample_count=32,
        max_entries=4,
    )

    assert samples.shape == (32, 1)
    assert jnp.all((samples >= 0) & (samples < 4))


def test_tensor_train_linear_matches_dense_forward_and_exposes_compression():
    left = jnp.asarray([[1.0, -0.5], [0.25, 2.0]], dtype=jnp.float32)
    right = jnp.asarray([[2.0, 0.0], [1.0, 1.5]], dtype=jnp.float32)
    weight = jnp.asarray(np.kron(np.asarray(left), np.asarray(right)))
    bias = jnp.asarray([0.1, -0.2, 0.3, 0.4], dtype=jnp.float32)
    layer = tt.TensorTrainLinear.from_dense(
        weight,
        (2, 2),
        (2, 2),
        bias=bias,
        max_ranks=1,
        relative_tolerance=0.0,
        max_dense_entries=16,
    )
    inputs = jnp.asarray(
        [[1.0, 2.0, 3.0, 4.0], [-1.0, 0.5, 2.0, -0.25]], dtype=jnp.float32
    )

    assert jnp.allclose(layer(inputs), inputs @ weight.T + bias, atol=2e-5)
    assert layer.compression_evidence.bound_satisfied
    assert layer.compression_evidence.measured_frobenius_error < 2e-5
    assert layer.compression_evidence.rounding.output_ranks == (1,)
