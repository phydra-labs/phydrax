#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._operators import AbstractLinearOperator, FunctionLinearOperator
from ._spaces import AbstractVectorSpace, DualSpace


class ConstraintMap(StrictModule, NonTrainableState):
    """Affine parameterization ``u = P z + g`` over declared vector spaces."""

    full_space: AbstractVectorSpace
    reduced_space: AbstractVectorSpace
    prolongation: AbstractLinearOperator
    dual_pullback: AbstractLinearOperator
    constraint_id: str = eqx.field(static=True)

    def __init__(
        self,
        full_space: AbstractVectorSpace,
        reduced_space: AbstractVectorSpace,
        prolongation: AbstractLinearOperator,
        /,
        *,
        constraint_id: str | None = None,
    ):
        if not isinstance(full_space, AbstractVectorSpace) or not isinstance(
            reduced_space, AbstractVectorSpace
        ):
            raise TypeError("Constraint spaces must be AbstractVectorSpace values.")
        if not isinstance(prolongation, AbstractLinearOperator):
            raise TypeError("prolongation must be an AbstractLinearOperator.")
        if not prolongation.source.compatible(
            reduced_space
        ) or not prolongation.target.compatible(full_space):
            raise ValueError("Constraint prolongation must map reduced to full space.")
        full_dual = DualSpace(full_space)
        reduced_dual = DualSpace(reduced_space)
        pullback = FunctionLinearOperator(
            lambda value: prolongation.transpose_mv(value),
            source=full_dual,
            target=reduced_dual,
            transpose_action=lambda value: prolongation.mv(value),
            operator_id=canonical_fingerprint(
                {
                    "kind": "constraint-dual-pullback",
                    "prolongation": prolongation.operator_id,
                }
            ),
        )
        resolved_id = (
            canonical_fingerprint(
                {
                    "kind": "constraint-map",
                    "full_space": full_space.space_id,
                    "reduced_space": reduced_space.space_id,
                    "prolongation": prolongation.operator_id,
                }
            )
            if constraint_id is None
            else str(constraint_id)
        )
        if not resolved_id:
            raise ValueError("constraint_id must be non-empty.")
        self.full_space = full_space
        self.reduced_space = reduced_space
        self.prolongation = prolongation
        self.dual_pullback = pullback
        self.constraint_id = resolved_id

    def expand(self, reduced: object, lift: object, /):
        reduced_ = self.reduced_space.validate(reduced)
        lift_ = self.full_space.validate(lift)
        return self.full_space.validate(
            jax.tree.map(
                lambda correction, offset: correction + offset,
                self.prolongation.mv(reduced_),
                lift_,
            )
        )

    def homogeneous_correction(self, reduced: Array, /):
        return self.prolongation.mv(self.reduced_space.validate(reduced))

    def pullback_dual(self, residual, /):
        return self.dual_pullback.mv(residual)

    def reduce_vector(self, residual, /):
        return self.prolongation.adjoint_mv(self.full_space.validate(residual))


__all__ = ["ConstraintMap"]
