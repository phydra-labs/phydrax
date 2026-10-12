from __future__ import annotations

import jax.numpy as jnp
import pytest

import phydrax.linalg as la


@pytest.mark.parametrize("workspace", [0, -1])
def test_materialization_workspace_limit_is_positive(workspace: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        la.MaterializationPolicy(max_workspace_bytes=workspace)


def test_full_materialization_admits_workspace_separately_from_output() -> None:
    operator = la.DenseLinearOperator(jnp.eye(3, dtype=jnp.float64))
    policy = la.MaterializationPolicy(max_entries=9, max_bytes=72, max_workspace_bytes=71)
    with pytest.raises(la.LinearCapabilityError, match="workspace requires 72 bytes"):
        la.materialize(operator, policy)


def _plan_identity(policy: la.MaterializationPolicy, kind: str) -> str:
    operator = la.DiagonalLinearOperator(jnp.asarray([1.0, 2.0], dtype=jnp.float64))
    if kind == "solve":
        return la.plan(
            la.LinearSystem(operator), la.LinearSolvePolicy(materialization=policy)
        ).plan_id
    if kind == "constraint":
        return la.ConstraintOperatorPlan(operator, materialization=policy).plan_id
    if kind == "svd":
        return la.svd.plan_svd(
            la.svd.SVDProblem(operator), la.svd.SVDSolvePolicy(materialization=policy)
        ).plan_id
    if kind == "eigen":
        return la.eigen.plan_eigensolve(
            la.eigen.Eigenproblem(operator),
            la.eigen.EigenSolvePolicy(
                la.eigen.DenseEigh(), count=2, materialization=policy
            ),
        ).plan_id
    raise ValueError("Unknown test plan")


@pytest.mark.parametrize("kind", ["solve", "constraint", "svd", "eigen"])
def test_workspace_budget_participates_in_plan_identity(kind: str) -> None:
    first = la.MaterializationPolicy(max_workspace_bytes=1024)
    repeated = la.MaterializationPolicy(max_workspace_bytes=1024)
    changed = la.MaterializationPolicy(max_workspace_bytes=2048)
    assert _plan_identity(first, kind) == _plan_identity(repeated, kind)
    assert _plan_identity(first, kind) != _plan_identity(changed, kind)
