#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


Q = phx.operators.quantum
X = jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex128)
Z = jnp.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=jnp.complex128)


def test_hilbert_register_layout_preserves_explicit_factor_order():
    layout = Q.HilbertRegisterLayout(("qubit", "qutrit", "mode"), (2, 3, 4))

    assert layout.dimension == 24
    assert layout.target_indices(("mode", "qubit")) == (2, 0)
    assert layout.target_dimension(("mode", "qubit")) == 8
    with pytest.raises(KeyError, match="Unknown Hilbert-register wire"):
        layout.wire_index("missing")
    with pytest.raises(ValueError, match="unique"):
        Q.HilbertRegisterLayout(("a", "a"), (2, 2))
    with pytest.raises(ValueError, match="positive"):
        Q.HilbertRegisterLayout(("a",), (0,))


def test_local_unitary_matches_global_embedding_and_target_order():
    layout = Q.HilbertRegisterLayout(("a", "b", "c"), (2, 3, 2))
    state = jnp.arange(1, 13, dtype=float).astype(jnp.complex128)
    state = state / jnp.linalg.norm(state)

    local_c = Q.apply_local_unitary_to_state(layout, X, ("c",), state)
    expected_c = jnp.kron(jnp.eye(6), X) @ state
    assert jnp.allclose(local_c, expected_c)

    local_ca = Q.apply_local_unitary_to_state(
        layout,
        jnp.kron(X, Z),
        ("c", "a"),
        state,
    )
    expected_ca = jnp.kron(jnp.kron(Z, jnp.eye(3)), X) @ state
    assert jnp.allclose(local_ca, expected_ca)


def test_local_unitary_supports_state_batches_and_density_conjugation():
    layout = Q.HilbertRegisterLayout(("a", "b"), (2, 2))
    states = jnp.stack(
        (
            jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.complex128),
            jnp.asarray([0.0, 0.0, 1.0, 0.0], dtype=jnp.complex128),
        )
    )
    transformed = jax.jit(
        lambda value: Q.apply_local_unitary_to_state(layout, X, ("b",), value)
    )(states)
    global_x = jnp.kron(jnp.eye(2), X)
    assert jnp.allclose(transformed, jax.vmap(lambda value: global_x @ value)(states))

    density = jnp.outer(states[1], jnp.conj(states[1]))
    local_density = Q.conjugate_local_density(layout, X, ("b",), density)
    expected_density = global_x @ density @ jnp.conj(global_x.T)
    assert jnp.allclose(local_density, expected_density)


def test_local_kraus_channel_is_trace_preserving_without_global_superoperator():
    layout = Q.HilbertRegisterLayout(("a", "b"), (2, 2))
    gamma = jnp.asarray(0.3)
    kraus = jnp.stack(
        (
            jnp.asarray(
                [[1.0, 0.0], [0.0, jnp.sqrt(1.0 - gamma)]],
                dtype=jnp.complex128,
            ),
            jnp.asarray(
                [[0.0, jnp.sqrt(gamma)], [0.0, 0.0]],
                dtype=jnp.complex128,
            ),
        )
    )
    excited_first = jnp.asarray([0.0, 0.0, 1.0, 0.0], dtype=jnp.complex128)
    density = jnp.outer(excited_first, jnp.conj(excited_first))

    result = Q.apply_local_kraus_to_density(layout, kraus, ("a",), density)

    expected = jnp.diag(jnp.asarray([gamma, 0.0, 1.0 - gamma, 0.0]))
    assert Q.kraus_trace_preservation_residual(kraus) < 1e-12
    assert jnp.allclose(result, expected)
    assert jnp.allclose(jnp.trace(result), 1.0)


def test_local_unitary_is_jittable_vmappable_and_real_objective_differentiable():
    layout = Q.HilbertRegisterLayout(("q",), (2,))
    state = jnp.asarray([1.0, 0.0], dtype=jnp.complex128)

    def objective(theta):
        unitary = jnp.diag(jnp.asarray([jnp.exp(1j * theta), 1.0], dtype=jnp.complex128))
        final = Q.apply_local_unitary_to_state(layout, unitary, ("q",), state)
        return jnp.real(final[0])

    theta = jnp.asarray(0.4)
    assert jnp.allclose(jax.jit(objective)(theta), jnp.cos(theta))
    assert jnp.allclose(jax.grad(objective)(theta), -jnp.sin(theta))
    assert jnp.allclose(
        jax.vmap(objective)(jnp.asarray([0.0, 0.4])),
        jnp.cos(jnp.asarray([0.0, 0.4])),
    )


def test_local_operations_reject_ambiguous_shapes_and_dtypes():
    layout = Q.HilbertRegisterLayout(("a", "b"), (2, 3))
    state = jnp.ones((6,), dtype=jnp.complex128) / jnp.sqrt(6.0)

    with pytest.raises(ValueError, match="dimension does not match"):
        Q.apply_local_unitary_to_state(layout, X, ("b",), state)
    with pytest.raises(ValueError, match="unique"):
        Q.apply_local_unitary_to_state(
            layout, jnp.eye(4, dtype=complex), ("a", "a"), state
        )
    with pytest.raises(TypeError, match="dtypes must match"):
        Q.apply_local_unitary_to_state(
            layout,
            X.astype(jnp.complex64),
            ("a",),
            state,
        )
    with pytest.raises(TypeError, match="complex"):
        Q.LocalUnitaryOperation(jnp.eye(2), ("a",))
    with pytest.raises(ValueError, match="K, dT, dT"):
        Q.LocalKrausChannelOperation(jnp.eye(2, dtype=complex), ("a",))
    with pytest.raises(ValueError, match="density-matrix"):
        Q.QuantumProgram(
            layout,
            (Q.LocalKrausChannelOperation(jnp.eye(2, dtype=complex)[None], ("a",)),),
            state_kind="state-vector",
        )
