#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ._operators import AbstractLinearOperator
from ._subspaces import NullspacePolicy


LinearProblemKind: TypeAlias = Literal["linear-system", "least-squares", "minimum-norm"]


class AbstractLinearProblem(StrictModule):
    """Mathematical problem independent of numerical solver choice."""

    operator: AbstractLinearOperator
    problem_id: str = eqx.field(static=True)
    nullspace_policy: NullspacePolicy | None

    @property
    @abc.abstractmethod
    def kind(self) -> LinearProblemKind:
        raise NotImplementedError


class LinearSystem(AbstractLinearProblem):
    """Exact square problem ``A x = b``."""

    nullspace_policy: NullspacePolicy | None

    def __init__(
        self,
        operator: AbstractLinearOperator,
        /,
        *,
        nullspace_policy: NullspacePolicy | None = None,
        problem_id: str | None = None,
    ):
        if not isinstance(operator, AbstractLinearOperator):
            raise TypeError("operator must be an AbstractLinearOperator.")
        if operator.source.size != operator.target.size:
            raise ValueError("LinearSystem requires equal source and target dimensions.")
        _validate_nullspace(nullspace_policy, operator)
        self.operator = operator
        self.nullspace_policy = nullspace_policy
        self.problem_id = _problem_id(
            problem_id,
            self.kind,
            operator.operator_id,
            nullspace_policy=nullspace_policy,
        )

    @property
    def kind(self) -> LinearProblemKind:
        return "linear-system"


class LeastSquaresProblem(AbstractLinearProblem):
    """Weighted residual minimization with optional zero-target regularization."""

    weights: Array | None
    regularizer: AbstractLinearOperator | None
    nullspace_policy: NullspacePolicy | None

    def __init__(
        self,
        operator: AbstractLinearOperator,
        /,
        *,
        weights: ArrayLike | None = None,
        regularizer: AbstractLinearOperator | None = None,
        nullspace_policy: NullspacePolicy | None = None,
        problem_id: str | None = None,
    ):
        if not isinstance(operator, AbstractLinearOperator):
            raise TypeError("operator must be an AbstractLinearOperator.")
        if weights is None:
            weights_ = None
        else:
            weights_ = jnp.asarray(weights)
            if not jnp.issubdtype(weights_.dtype, jnp.inexact):
                weights_ = weights_.astype(float)
            if jnp.issubdtype(weights_.dtype, jnp.complexfloating):
                raise TypeError("Least-squares weights must be real-valued.")
            weights_ = eqx.error_if(
                weights_,
                jnp.any(~jnp.isfinite(weights_)) | jnp.any(weights_ < 0.0),
                "Least-squares weights must be finite and non-negative.",
            )
        if regularizer is not None:
            if not isinstance(regularizer, AbstractLinearOperator):
                raise TypeError("regularizer must be an AbstractLinearOperator or None.")
            if not regularizer.source.compatible(operator.source):
                raise ValueError("regularizer source must match the problem source.")
            if regularizer.batch_shape != operator.batch_shape:
                raise ValueError("regularizer and operator batch shapes must match.")
        _validate_nullspace(nullspace_policy, operator)
        self.operator = operator
        self.weights = weights_
        self.regularizer = regularizer
        self.nullspace_policy = nullspace_policy
        regularizer_id = None if regularizer is None else regularizer.operator_id
        weights_structure: dict[str, object] | None
        weights_structure = (
            None
            if weights_ is None
            else {
                "shape": list(weights_.shape),
                "dtype": jnp.dtype(weights_.dtype).str,
            }
        )
        self.problem_id = _problem_id(
            problem_id,
            self.kind,
            operator.operator_id,
            regularizer_id,
            weights_structure,
            nullspace_policy,
        )

    @property
    def kind(self) -> LinearProblemKind:
        return "least-squares"


class MinimumNormProblem(AbstractLinearProblem):
    """Minimum source-norm solution subject to ``A x = b``."""

    nullspace_policy: NullspacePolicy | None

    def __init__(
        self,
        operator: AbstractLinearOperator,
        /,
        *,
        problem_id: str | None = None,
        nullspace_policy: NullspacePolicy | None = None,
    ):
        if not isinstance(operator, AbstractLinearOperator):
            raise TypeError("operator must be an AbstractLinearOperator.")
        if operator.source.size < operator.target.size:
            raise ValueError(
                "MinimumNormProblem requires source dimension at least target dimension."
            )
        _validate_nullspace(nullspace_policy, operator)
        self.operator = operator
        self.nullspace_policy = nullspace_policy
        self.problem_id = _problem_id(
            problem_id,
            self.kind,
            operator.operator_id,
            nullspace_policy=nullspace_policy,
        )

    @property
    def kind(self) -> LinearProblemKind:
        return "minimum-norm"


def _problem_id(
    value: str | None,
    kind: LinearProblemKind,
    operator_id: str,
    regularizer_id: str | None = None,
    weights_structure: dict[str, object] | None = None,
    nullspace_policy: NullspacePolicy | None = None,
) -> str:
    if value is None:
        return canonical_fingerprint(
            {
                "kind": kind,
                "operator": operator_id,
                "regularizer": regularizer_id,
                "weights": weights_structure,
                "nullspace": _nullspace_payload(nullspace_policy),
            }
        )
    identifier = str(value)
    if not identifier:
        raise ValueError("problem_id must be non-empty.")
    return identifier


def _validate_nullspace(
    policy: NullspacePolicy | None,
    operator: AbstractLinearOperator,
    /,
) -> None:
    if policy is None:
        return
    if not isinstance(policy, NullspacePolicy):
        raise TypeError("nullspace_policy must be a NullspacePolicy or None.")
    if operator.batch_shape:
        raise ValueError("Declared nullspaces currently require an unbatched operator.")
    if policy.right is not None and not policy.right.space.compatible(operator.source):
        raise ValueError("Right nullspace must use the operator source space.")
    if policy.left is not None and not policy.left.space.compatible(operator.target):
        raise ValueError("Left nullspace must use the operator target space.")


def _nullspace_payload(policy: NullspacePolicy | None, /) -> dict[str, object] | None:
    if policy is None:
        return None
    return {
        "right": None if policy.right is None else policy.right.subspace_id,
        "left": None if policy.left is None else policy.left.subspace_id,
        "compatibility": policy.compatibility,
        "gauge": policy.gauge,
        "certificate": (
            None if policy.certificate is None else policy.certificate.structure_id
        ),
    }


def _problem_structure(problem: AbstractLinearProblem, /) -> str:
    """Fingerprint problem structure independently of an overridden problem ID."""
    regularizer = (
        problem.regularizer if isinstance(problem, LeastSquaresProblem) else None
    )
    weights = problem.weights if isinstance(problem, LeastSquaresProblem) else None
    nullspace_policy = problem.nullspace_policy
    return canonical_fingerprint(
        {
            "kind": problem.kind,
            "operator": problem.operator.operator_id,
            "regularizer": None if regularizer is None else regularizer.operator_id,
            "weights": (
                None
                if weights is None
                else {
                    "shape": list(weights.shape),
                    "dtype": jnp.dtype(weights.dtype).str,
                }
            ),
            "nullspace": _nullspace_payload(nullspace_policy),
        }
    )


__all__ = [
    "AbstractLinearProblem",
    "LeastSquaresProblem",
    "LinearProblemKind",
    "LinearSystem",
    "MinimumNormProblem",
]
