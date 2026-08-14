import jax.numpy as jnp

import phydrax as phx


def test_tensor_exponential_and_chen_product_match_piecewise_linear_signature():
    left = jnp.asarray([1.0, 0.0])
    right = jnp.asarray([0.0, 1.0])
    left_signature = phx.stochastic.tensor_exponential(left, 3)
    right_signature = phx.stochastic.tensor_exponential(right, 3)
    composed = phx.stochastic.chen_multiply(left_signature, right_signature)
    piecewise = phx.stochastic.piecewise_linear_signature(jnp.stack((left, right)), 3)

    assert jnp.allclose(left_signature[1], 0.5 * jnp.outer(left, left))
    assert jnp.allclose(
        left_signature[2],
        jnp.einsum("i,j,k->ijk", left, left, left) / 6.0,
    )
    assert all(
        jnp.allclose(composed_level, piecewise_level)
        for composed_level, piecewise_level in zip(composed, piecewise)
    )
    assert jnp.allclose(composed[1][0, 1], 1.0)
    assert jnp.allclose(composed[1][1, 0], 0.0)


def test_piecewise_linear_signature_promotes_integer_increments():
    increments = jnp.asarray([[1, 0], [0, 1]], dtype=jnp.int32)
    signature = phx.stochastic.piecewise_linear_signature(increments, 3)

    assert all(jnp.issubdtype(level.dtype, jnp.inexact) for level in signature)
    assert jnp.allclose(signature[0], jnp.asarray([1.0, 1.0]))
    assert jnp.allclose(signature[1][0, 1], 1.0)
    assert jnp.allclose(signature[2][0, 0, 1], 0.5)


def test_lyndon_basis_tensor_log_conversion_recovers_bch_coefficients():
    signature = phx.stochastic.piecewise_linear_signature(
        jnp.asarray([[1.0, 0.0], [0.0, 1.0]]), 3
    )
    tensor_log = phx.stochastic.tensor_logarithm(signature)
    basis = phx.stochastic.PrimitiveBasis(2, 3)
    primitive = basis.tensor_to_primitive(tensor_log)
    reconstructed = basis.primitive_to_tensor(primitive)

    assert basis.words == ((0,), (1,), (0, 1), (0, 0, 1), (0, 1, 1))
    assert basis.children == (None, None, (0, 1), (0, 2), (2, 1))
    assert jnp.allclose(
        primitive,
        jnp.asarray([1.0, 1.0, 0.5, 1.0 / 12.0, 1.0 / 12.0]),
    )
    assert all(
        jnp.allclose(actual, expected)
        for actual, expected in zip(reconstructed, tensor_log)
    )
    bracket_expansion = dict(basis.word_expansions[2])
    assert bracket_expansion == {(0, 1): 1, (1, 0): -1}


def test_log_signature_control_aggregates_fine_knots_and_records_provenance():
    fine_times = jnp.linspace(0.0, 1.0, 5)
    values = jnp.asarray([[0.0, 0.0], [0.2, -0.1], [0.1, 0.4], [0.7, 0.5], [0.6, 1.0]])
    control = phx.stochastic.LogSignatureControl.from_values(
        fine_times,
        values,
        depth=3,
        coarse_indices=(0, 2, 4),
        joint_time=True,
        source_id="deterministic-sample-7",
    )
    repeated = phx.stochastic.LogSignatureControl.from_values(
        fine_times,
        values,
        depth=3,
        coarse_times=jnp.asarray([0.0, 0.5, 1.0]),
        joint_time=True,
        source_id="deterministic-sample-7",
    )
    augmented_values = jnp.concatenate((fine_times[:, None], values), axis=-1)
    exact = phx.stochastic.piecewise_linear_signature(
        jnp.diff(augmented_values, axis=0), 3
    )

    assert control.depth == 3
    assert control.dimension == 3
    assert control.source_dimension == 2
    assert control.joint_time
    assert control.source_id == "deterministic-sample-7"
    assert control.control_id == repeated.control_id
    assert all(
        jnp.allclose(actual, expected)
        for actual, expected in zip(control.terminal_signature, exact)
    )
    assert all(
        jnp.allclose(actual, expected)
        for actual, expected in zip(control.levels, repeated.levels)
    )


def test_depth_four_log_signature_control_promotes_integer_values():
    times = jnp.asarray([0, 1, 2], dtype=jnp.int32)
    values = jnp.asarray([[0, 0], [1, 0], [1, 1]], dtype=jnp.int32)
    control = phx.stochastic.LogSignatureControl.from_values(
        times,
        values,
        depth=4,
        coarse_indices=(0, 2),
        source_id="integer-depth-four",
    )
    expected = phx.stochastic.piecewise_linear_signature(jnp.diff(values, axis=0), 4)
    tensor_log = phx.stochastic.tensor_logarithm(control.terminal_signature)
    reconstructed = control.primitive_basis.primitive_to_tensor(
        control.primitive_basis.tensor_to_primitive(tensor_log)
    )

    assert control.depth == 4
    assert all(jnp.issubdtype(level.dtype, jnp.inexact) for level in control.levels)
    assert all(
        jnp.allclose(actual, target)
        for actual, target in zip(control.terminal_signature, expected)
    )
    assert all(
        jnp.allclose(actual, target) for actual, target in zip(reconstructed, tensor_log)
    )
