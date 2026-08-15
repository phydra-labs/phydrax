#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from .._operators import AbstractLinearOperator
from .._subspaces import LinearSubspace


EigenproblemKind: TypeAlias = Literal["standard", "generalized"]


class Eigenproblem(StrictModule):
    """Certified unbatched self-adjoint eigenproblem ``A x = lambda x``."""

    operator: AbstractLinearOperator
    constraints: LinearSubspace | None
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        operator: AbstractLinearOperator,
        /,
        *,
        constraints: LinearSubspace | None = None,
        problem_id: str | None = None,
    ):
        _validate_self_adjoint_endomorphism(operator, "operator")
        _validate_constraints(constraints, operator)
        self.operator = operator
        self.constraints = constraints
        self.problem_id = _problem_id(
            problem_id,
            kind="standard",
            operator_id=operator.operator_id,
            metric_id=None,
            constraints=constraints,
        )

    @property
    def kind(self) -> EigenproblemKind:
        return "standard"

    @property
    def dimension(self) -> int:
        return self.operator.source.size


class GeneralizedEigenproblem(StrictModule):
    """Certified generalized self-adjoint eigenproblem ``A x = lambda B x``."""

    operator: AbstractLinearOperator
    metric_operator: AbstractLinearOperator
    constraints: LinearSubspace | None
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        operator: AbstractLinearOperator,
        metric_operator: AbstractLinearOperator,
        /,
        *,
        constraints: LinearSubspace | None = None,
        problem_id: str | None = None,
    ):
        _validate_self_adjoint_endomorphism(operator, "operator")
        _validate_self_adjoint_endomorphism(metric_operator, "metric_operator")
        if not metric_operator.properties.certifies("positive_definite"):
            raise ValueError(
                "GeneralizedEigenproblem requires a certified positive-definite "
                "metric_operator."
            )
        if not metric_operator.source.compatible(
            operator.source
        ) or not metric_operator.target.compatible(operator.target):
            raise ValueError(
                "The generalized metric_operator must act on the operator vector space."
            )
        _validate_constraints(constraints, operator)
        self.operator = operator
        self.metric_operator = metric_operator
        self.constraints = constraints
        self.problem_id = _problem_id(
            problem_id,
            kind="generalized",
            operator_id=operator.operator_id,
            metric_id=metric_operator.operator_id,
            constraints=constraints,
        )

    @property
    def kind(self) -> EigenproblemKind:
        return "generalized"

    @property
    def dimension(self) -> int:
        return self.operator.source.size


def _validate_self_adjoint_endomorphism(
    operator: AbstractLinearOperator,
    name: str,
    /,
) -> None:
    if not isinstance(operator, AbstractLinearOperator):
        raise TypeError(f"{name} must be an AbstractLinearOperator.")
    if operator.batch_shape:
        raise ValueError(f"Eigenproblem {name} must be unbatched.")
    if not operator.source.compatible(operator.target):
        raise ValueError(f"Eigenproblem {name} must be an endomorphism.")
    if not operator.properties.certifies("self_adjoint"):
        raise ValueError(
            f"Eigenproblem {name} must carry certified self-adjoint evidence."
        )


def _validate_constraints(
    constraints: LinearSubspace | None,
    operator: AbstractLinearOperator,
    /,
) -> None:
    if constraints is None:
        return
    if not isinstance(constraints, LinearSubspace):
        raise TypeError("constraints must be a LinearSubspace or None.")
    if not constraints.space.compatible(operator.source):
        raise ValueError("Constraint space must match the eigenproblem vector space.")


def _problem_id(
    value: str | None,
    /,
    *,
    kind: EigenproblemKind,
    operator_id: str,
    metric_id: str | None,
    constraints: LinearSubspace | None,
) -> str:
    if value is None:
        return canonical_fingerprint(
            {
                "kind": f"{kind}-eigenproblem",
                "operator": operator_id,
                "metric": metric_id,
                "constraints": (None if constraints is None else constraints.subspace_id),
                "constraint_capacity": (
                    0 if constraints is None else constraints.capacity
                ),
            }
        )
    identifier = str(value)
    if not identifier:
        raise ValueError("problem_id must be non-empty.")
    return identifier


EigenproblemLike: TypeAlias = Eigenproblem | GeneralizedEigenproblem


__all__ = [
    "Eigenproblem",
    "EigenproblemKind",
    "GeneralizedEigenproblem",
]
