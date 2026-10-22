"""Rank-deficient DIIS is a constrained minimum-norm problem, not failed LU."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax.chemistry.electronic_structure import _molecular_hf as hf
from phydrax.linalg import (
    LeastSquaresProblem,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSolveStatus,
)


@pytest.mark.parametrize("kind", ("zero", "identical", "dependent", "complex"))
@pytest.mark.parametrize("scale", (1.0e-100, 1.0, 1.0e100))
def test_rank_deficient_diis_preserves_affine_minimum_norm_semantics(
    kind: str,
    scale: float,
) -> None:
    focks = [jnp.eye(2, dtype=jnp.float64) * value for value in (1.0, 4.0, 9.0)]
    pattern = jnp.asarray([[0.0, 1.0], [-1.0, 0.0]], dtype=jnp.float64)
    if kind == "zero":
        errors = [jnp.zeros_like(pattern) for _ in focks]
        expected = np.full((3,), 1.0 / 3.0, dtype=np.float64)
        expected_rank = 0
    elif kind == "identical":
        errors = [scale * pattern for _ in focks]
        expected = np.full((3,), 1.0 / 3.0, dtype=np.float64)
        expected_rank = 0
    elif kind in ("dependent", "complex"):
        if kind == "complex":
            pattern = pattern.astype(jnp.complex128) * (1.0 + 2.0j)
        errors = [scale * multiplier * pattern for multiplier in (1.0, 2.0, 3.0)]
        expected = np.asarray([4.0 / 3.0, 1.0 / 3.0, -2.0 / 3.0], dtype=np.float64)
        expected_rank = 1
    else:
        raise ValueError(kind)
    fock, evidence = hf._diis_extrapolate(focks, errors, 8)

    assert bool(evidence.successful)
    assert int(evidence.linear_solve.status) == int(LinearSolveStatus.SUCCESS)
    assert int(evidence.linear_solve.diagnostics.rank) == expected_rank
    assert float(evidence.constraint_residual) <= float(evidence.constraint_tolerance)
    np.testing.assert_allclose(
        evidence.coefficients, expected, rtol=2.0e-13, atol=2.0e-14
    )
    np.testing.assert_allclose(
        np.sum(np.asarray(evidence.coefficients)), 1.0, atol=1.0e-14
    )
    np.testing.assert_allclose(
        fock, np.dot(expected, [1.0, 4.0, 9.0]) * np.eye(2), atol=2.0e-13
    )


def test_diis_native_failure_retains_status_rank_and_constraint_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    focks = [jnp.eye(2, dtype=jnp.float64) for _ in range(2)]
    errors = [jnp.zeros_like(value) for value in focks]
    _, original = hf._diis_extrapolate(focks, errors, 8)
    failed = eqx.tree_at(
        lambda result: result.status,
        original.linear_solve,
        jnp.asarray(int(LinearSolveStatus.SINGULAR), dtype=jnp.int32),
    )

    def fail_solve(
        problem: LeastSquaresProblem,
        rhs: Array,
        *,
        policy: LinearSolvePolicy,
    ) -> LinearSolveResult:
        return failed

    monkeypatch.setattr(hf, "solve", fail_solve)
    with pytest.raises(hf._SCFDIISSolveError, match="native_status=4, rank=0") as error:
        hf._diis_extrapolate(focks, errors, 8)
    assert error.value.evidence.linear_solve is failed
    assert not bool(error.value.evidence.successful)
