#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


q = phx.operators.quantum


def test_basis_state_subspace_embeds_restricts_and_projects_without_one_hot_storage():
    layout = q.HilbertRegisterLayout(("q0", "c", "q1"), (2, 3, 2))
    subspace = q.basis_state_subspace(
        layout,
        ((0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 0, 1)),
    )
    logical = jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.complex128)
    physical = q.embed_quantum_subspace(subspace, logical)
    operator = jnp.arange(144.0).reshape((12, 12)).astype(jnp.complex128)

    assert subspace.logical_dimension == 4
    assert subspace.physical_dimension == 12
    assert jnp.count_nonzero(physical) == 4
    assert jnp.allclose(q.restrict_quantum_subspace(subspace, physical), logical)
    assert jnp.allclose(
        q.project_quantum_operator(operator, subspace),
        operator[subspace.basis_indices[:, None], subspace.basis_indices[None, :]],
    )


def test_dense_subspace_supports_batched_states_and_mixed_projection():
    angle = jnp.asarray(0.31)
    isometry = jnp.asarray(
        [
            [jnp.cos(angle), 0.0],
            [jnp.sin(angle), 0.0],
            [0.0, 1.0],
        ],
        dtype=jnp.complex128,
    )
    dense = q.DenseQuantumSubspace(isometry)
    basis = q.BasisStateSubspace(3, (0, 2))
    logical_batch = jnp.asarray([[1.0, 0.0], [0.0, 1.0]])
    physical_batch = jax.jit(q.embed_quantum_subspace)(dense, logical_batch)
    operator = jnp.asarray(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        dtype=jnp.complex128,
    )

    assert bool(dense.evidence.valid)
    assert jnp.allclose(q.restrict_quantum_subspace(dense, physical_batch), logical_batch)
    assert jnp.allclose(
        q.project_quantum_operator(operator, dense, basis),
        operator[basis.basis_indices] @ isometry,
    )


def test_quantum_subspaces_reject_invalid_indices_isometries_and_shapes():
    with pytest.raises(ValueError, match="unique"):
        q.BasisStateSubspace(4, (1, 1))
    with pytest.raises(ValueError, match="within"):
        q.BasisStateSubspace(4, (0, 4))
    invalid = q.DenseQuantumSubspace(jnp.asarray([[1.0, 1.0], [0.0, 0.0]]))
    assert not bool(invalid.evidence.valid)

    layout = q.HilbertRegisterLayout(("a", "b"), (2, 3))
    with pytest.raises(ValueError, match="cover every"):
        q.basis_state_subspace(layout, ((0,),))
    with pytest.raises(ValueError, match="out-of-range"):
        q.basis_state_subspace(layout, ((0, 3),))
    with pytest.raises(ValueError, match="trailing shape"):
        q.project_quantum_operator(
            jnp.eye(3),
            q.BasisStateSubspace(4, (0, 1)),
        )
