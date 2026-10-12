# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Host sparse batches preserve solves, residual evidence, and provider reuse."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse.linalg as spla

import phydrax.linalg as la
from phydrax.linalg.backends import _jax_sparse as backend
from phydrax.sparse import EdgeRelation, SparseCoordinateOperator


def _problem() -> tuple[la.LinearSystem, np.ndarray]:
    matrix = np.asarray(
        [[4 + 0.3j, 0, 1 - 0.5j], [2 + 0.7j, 5 - 0.1j, 0], [0, 1 + 0.2j, 3 + 0.4j]]
    )
    rows, columns = np.nonzero(matrix)
    operator = SparseCoordinateOperator(
        EdgeRelation(
            columns.astype(np.int32), rows.astype(np.int32), source_size=3, target_size=3
        ),
        jnp.asarray(matrix[rows, columns]),
        source=la.ArraySpace((3,), dtype=jnp.complex128),
        target=la.ArraySpace((3,), dtype=jnp.complex128),
    )
    return la.LinearSystem(operator), matrix


def test_superlu_multi_rhs_reuses_one_factor_and_preserves_transformed_solves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem, matrix = _problem()
    calls = []
    original = spla.splu

    def counted(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(spla, "splu", counted)
    prepared = la.prepare(
        problem,
        la.LinearSolvePolicy(
            la.SparseLU(), differentiation=la.DifferentiationPolicy("none")
        ),
    )
    rhs = np.arange(12, dtype=np.float64).reshape(3, 4) + 0.5j
    layout = la.RHSLayout((4,))
    for operation, reference_matrix in (
        (la.solve, matrix),
        (la.solve_transpose, matrix.T),
        (la.solve_adjoint, matrix.conj().T),
    ):
        result = operation(prepared, jnp.asarray(rhs), rhs_layout=layout)
        assert bool(jnp.all(result.successful))
        np.testing.assert_allclose(
            result.value, np.linalg.solve(reference_matrix, rhs), rtol=1e-12, atol=1e-12
        )
    assert len(calls) == 1
    # The provider owns its factors; no duplicate CSR arrays are device leaves.
    assert (
        sum(
            leaf.nbytes
            for leaf in jax.tree.leaves(prepared.state)
            if isinstance(leaf, jax.Array)
        )
        == 0
    )


def test_host_failed_numeric_solve_cannot_be_reported_as_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem, _ = _problem()
    prepared = la.prepare(
        problem,
        la.LinearSolvePolicy(
            la.SparseLU(), differentiation=la.DifferentiationPolicy("none")
        ),
    )

    def failed(
        provider: str,
        factor: Any,
        rhs: np.ndarray,
        /,
        *,
        transpose: bool = False,
        adjoint: bool = False,
    ) -> np.ndarray:
        del provider, factor, transpose, adjoint
        return np.zeros_like(rhs)

    monkeypatch.setattr(backend, "_host_solve", failed)
    result = la.solve(
        prepared, jnp.ones((3, 2), dtype=jnp.complex128), rhs_layout=la.RHSLayout((2,))
    )
    assert not bool(jnp.any(result.successful))
    np.testing.assert_array_equal(
        result.status, int(la.LinearSolveStatus.RESIDUAL_TOO_LARGE)
    )


@pytest.mark.parametrize("provider", ("umfpack", "spqr", "cholmod"))
def test_installed_optional_provider_multi_rhs(provider: str) -> None:
    module = {
        "umfpack": "scikits.umfpack",
        "spqr": "sparseqr",
        "cholmod": "sksparse.cholmod",
    }[provider]
    pytest.importorskip(module)
    import scipy.sparse as sp

    matrix = np.asarray([[4.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 3.0]])
    factor = backend._host_factor(provider, sp.csc_matrix(matrix), np.arange(3))
    rhs = np.arange(12, dtype=np.float64).reshape(3, 4)
    for transpose, adjoint in ((False, False), (True, False), (False, True)):
        solution = backend._host_solve(
            provider, factor, rhs, transpose=transpose, adjoint=adjoint
        )
        np.testing.assert_allclose(
            solution, np.linalg.solve(matrix, rhs), rtol=1e-12, atol=1e-12
        )
