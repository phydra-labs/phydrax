#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from .._fingerprint import canonical_fingerprint
from ._operators import (
    AbstractLinearOperator,
    adjoint,
    BlockLinearOperator,
    IdentityLinearOperator,
    ScaledLinearOperator,
)
from ._preconditioners import AbstractPreconditioner
from ._problems import LinearSystem
from ._properties import OperatorProperties
from ._spaces import BlockSpace
from ._structured_operators import SchurComplementLinearOperator
from ._subspaces import NullspacePolicy


def saddle_point_operator(
    primal_operator: AbstractLinearOperator,
    constraint_operator: AbstractLinearOperator,
    stabilization: AbstractLinearOperator | None = None,
    /,
    *,
    properties: OperatorProperties | None = None,
    operator_id: str | None = None,
) -> BlockLinearOperator:
    """Assemble ``[[A, B*], [B, -C]]`` over an explicit block space."""
    _validate_blocks(primal_operator, constraint_operator, stabilization)
    block_space = BlockSpace((primal_operator.source, constraint_operator.target))
    inferred_self_adjoint = primal_operator.properties.self_adjoint and (
        stabilization is None or stabilization.properties.self_adjoint
    )
    certified_self_adjoint = primal_operator.properties.certifies("self_adjoint") and (
        stabilization is None or stabilization.properties.certifies("self_adjoint")
    )
    properties_ = (
        OperatorProperties(
            self_adjoint=inferred_self_adjoint,
            evidence=({"self_adjoint": "transformed"} if certified_self_adjoint else {}),
        )
        if properties is None
        else properties
    )
    identifier = (
        canonical_fingerprint(
            {
                "kind": "saddle-point",
                "primal": primal_operator.operator_id,
                "constraint": constraint_operator.operator_id,
                "stabilization": None
                if stabilization is None
                else stabilization.operator_id,
            }
        )
        if operator_id is None
        else str(operator_id)
    )
    if not identifier:
        raise ValueError("operator_id must be non-empty.")
    lower_diagonal = None if stabilization is None else -stabilization
    return BlockLinearOperator(
        (
            (primal_operator, adjoint(constraint_operator)),
            (constraint_operator, lower_diagonal),
        ),
        source=block_space,
        target=block_space,
        properties=properties_,
        operator_id=identifier,
    )


def saddle_point_system(
    primal_operator: AbstractLinearOperator,
    constraint_operator: AbstractLinearOperator,
    stabilization: AbstractLinearOperator | None = None,
    /,
    *,
    properties: OperatorProperties | None = None,
    nullspace_policy: NullspacePolicy | None = None,
    operator_id: str | None = None,
    problem_id: str | None = None,
) -> LinearSystem:
    """Build a linear-system problem for a saddle-point operator."""
    operator = saddle_point_operator(
        primal_operator,
        constraint_operator,
        stabilization,
        properties=properties,
        operator_id=operator_id,
    )
    return LinearSystem(
        operator,
        nullspace_policy=nullspace_policy,
        problem_id=problem_id,
    )


def saddle_point_schur_complement(
    primal_operator: AbstractLinearOperator,
    constraint_operator: AbstractLinearOperator,
    inverse_action: AbstractPreconditioner,
    stabilization: AbstractLinearOperator | None = None,
    /,
    *,
    operator_id: str | None = None,
) -> SchurComplementLinearOperator:
    """Build the dual Schur complement ``-C - B A⁻¹ B*`` matrix-free."""
    _validate_blocks(primal_operator, constraint_operator, stabilization)
    if not isinstance(inverse_action, AbstractPreconditioner):
        raise TypeError("inverse_action must be an AbstractPreconditioner.")
    diagonal = (
        ScaledLinearOperator(IdentityLinearOperator(constraint_operator.target), 0.0)
        if stabilization is None
        else -stabilization
    )
    return SchurComplementLinearOperator(
        diagonal,
        constraint_operator,
        inverse_action,
        adjoint(constraint_operator),
        operator_id=operator_id,
    )


def _validate_blocks(
    primal_operator: AbstractLinearOperator,
    constraint_operator: AbstractLinearOperator,
    stabilization: AbstractLinearOperator | None,
    /,
) -> None:
    if not isinstance(primal_operator, AbstractLinearOperator) or not isinstance(
        constraint_operator, AbstractLinearOperator
    ):
        raise TypeError(
            "primal_operator and constraint_operator must be linear operators."
        )
    if not primal_operator.source.compatible(primal_operator.target):
        raise ValueError("primal_operator must be an endomorphism.")
    if not constraint_operator.source.compatible(primal_operator.source):
        raise ValueError("constraint_operator must map from the primal space.")
    if stabilization is not None:
        if not isinstance(stabilization, AbstractLinearOperator):
            raise TypeError("stabilization must be a linear operator or None.")
        if not stabilization.source.compatible(constraint_operator.target) or not (
            stabilization.target.compatible(constraint_operator.target)
        ):
            raise ValueError("stabilization must be an endomorphism on the dual space.")


__all__ = [
    "saddle_point_operator",
    "saddle_point_schur_complement",
    "saddle_point_system",
]
