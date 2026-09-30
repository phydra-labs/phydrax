#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import final, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import PyTree

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...exterior._complex import ComplexBoundary
from ...linalg import (
    AbstractLinearOperator,
    AbstractPreconditioner,
    AbstractPreconditionerBuilder,
    AdditiveSubspaceCorrectionBuilder,
    DiagonalLinearOperator,
    PreconditionerProperties,
    SubspaceCorrectionTerm,
)
from ...linalg._complexes import coordinate_operator, coordinate_space
from ...linalg._operators import adjoint, AdjointLinearOperator, IdentityLinearOperator
from ...sparse import EdgeRelation, SparseCoordinateOperator


if TYPE_CHECKING:
    from ._de_rham import FiniteElementDeRhamComplex


@final
class LowOrderAuxiliaryOperatorPlan(StrictModule, NonTrainableState):
    interpolation: AbstractLinearOperator
    anterpolation: AbstractLinearOperator
    multiplicity_weight: object
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        interpolation: AbstractLinearOperator,
        anterpolation: AbstractLinearOperator,
        multiplicity_weight: object,
        /,
    ) -> None:
        if not isinstance(interpolation, AbstractLinearOperator) or not isinstance(
            anterpolation, AbstractLinearOperator
        ):
            raise TypeError("Auxiliary transfers must be linear operators.")
        if not interpolation.source.compatible(
            anterpolation.target
        ) or not interpolation.target.compatible(anterpolation.source):
            raise ValueError(
                "Auxiliary interpolation/anterpolation spaces are incompatible."
            )
        weight = interpolation.source.validate(multiplicity_weight)
        leaves = jax.tree.leaves(weight)
        if any(bool(jnp.any(~jnp.isfinite(value) | (value <= 0.0))) for value in leaves):
            raise ValueError(
                "Auxiliary multiplicity weights must be positive and finite."
            )
        self.interpolation = interpolation
        self.anterpolation = anterpolation
        self.multiplicity_weight = weight
        self.plan_id = canonical_fingerprint(
            {
                "kind": "low-order-auxiliary-operator-plan",
                "interpolation": interpolation.operator_id,
                "anterpolation": anterpolation.operator_id,
                "multiplicity_weight": array_tree_fingerprint(weight),
                "high_space": interpolation.source.space_id,
                "low_space": interpolation.target.space_id,
            }
        )

    @classmethod
    def from_complex(
        cls,
        complex: FiniteElementDeRhamComplex,
        degree: int,
        /,
        *,
        boundary: ComplexBoundary = "absolute",
    ) -> LowOrderAuxiliaryOperatorPlan:
        """Prepare canonical low-order inclusion and its coordinate adjoint."""
        from ._de_rham import FiniteElementDeRhamComplex

        if not isinstance(complex, FiniteElementDeRhamComplex):
            raise TypeError("complex must be a FiniteElementDeRhamComplex.")
        if complex.order <= 1:
            raise ValueError(
                "A low-order auxiliary plan requires order greater than one."
            )
        low = FiniteElementDeRhamComplex(
            complex.mesh,
            family=complex.family,
            order=1,
            twist=complex.primal_twist,
        )

        def restriction(
            realization: FiniteElementDeRhamComplex,
        ) -> AbstractLinearOperator:
            full = coordinate_space(realization.hilbert_complex().space(degree))
            active = realization.active_indices(degree, boundary=boundary)
            reduced = coordinate_space(
                realization.hilbert_complex(boundary=boundary).space(degree)
            )
            relation = EdgeRelation(
                active,
                jnp.arange(active.size, dtype=jnp.int32),
                source_size=full.size,
                target_size=reduced.size,
            )
            return SparseCoordinateOperator(
                relation,
                jnp.ones((active.size,), dtype=jnp.float64),
                source=full,
                target=reduced,
                operator_id=f"{realization.realization_id}:auxiliary-restriction:{degree}:{boundary}",
            )

        inclusion = coordinate_operator(low.transfer(complex).maps[degree])
        prolongation = restriction(complex) @ inclusion @ adjoint(restriction(low))
        return cls(
            adjoint(prolongation),
            prolongation,
            jnp.ones((prolongation.target.size,), dtype=jnp.float64),
        )


@final
class LowOrderAuxiliaryPreconditioner(AbstractPreconditioner):
    plan: LowOrderAuxiliaryOperatorPlan
    low_order_preconditioner: AbstractPreconditioner

    def __init__(
        self,
        plan: LowOrderAuxiliaryOperatorPlan,
        low_order_preconditioner: AbstractPreconditioner,
        /,
    ) -> None:
        if not isinstance(plan, LowOrderAuxiliaryOperatorPlan) or not isinstance(
            low_order_preconditioner, AbstractPreconditioner
        ):
            raise TypeError("Auxiliary preconditioner inputs are invalid.")
        if not low_order_preconditioner.space.compatible(plan.interpolation.target):
            raise ValueError("Low-order preconditioner acts on the wrong space.")
        self.plan = plan
        self.low_order_preconditioner = low_order_preconditioner
        self.space = plan.interpolation.source
        self.properties = PreconditionerProperties(
            linear=low_order_preconditioner.properties.certifies("linear"),
            stationary=low_order_preconditioner.properties.certifies("stationary"),
            evidence={
                name: "transformed"
                for name in ("linear", "stationary")
                if low_order_preconditioner.properties.certifies(name)
            },
        )
        self.preconditioner_id = canonical_fingerprint(
            {
                "kind": "low-order-auxiliary-preconditioner",
                "plan": plan.plan_id,
                "inner": low_order_preconditioner.preconditioner_id,
            }
        )

    def apply(
        self,
        residual: PyTree,
        /,
        *,
        iteration: ArrayLike | None = None,
    ) -> PyTree[Array]:
        checked = self.space.validate(residual)
        weighted = jax.tree.map(
            lambda value, weight: value * weight,
            checked,
            self.plan.multiplicity_weight,
        )
        low_residual = self.plan.interpolation.mv(weighted)
        low_correction = self.low_order_preconditioner.apply(
            low_residual,
            iteration=iteration,
        )
        high_correction = self.plan.anterpolation.mv(low_correction)
        return self.space.validate(
            jax.tree.map(
                lambda value, weight: value * weight,
                high_correction,
                self.plan.multiplicity_weight,
            )
        )


def low_order_auxiliary_preconditioner_builder(
    plan: LowOrderAuxiliaryOperatorPlan,
    low_order_solver: AbstractPreconditioner | AbstractPreconditionerBuilder,
    /,
    *,
    properties: PreconditionerProperties | None = None,
    smoother: AbstractPreconditioner | AbstractPreconditionerBuilder | None = None,
) -> AdditiveSubspaceCorrectionBuilder:
    """Build the weighted auxiliary correction on the generic linalg substrate."""
    if not isinstance(plan, LowOrderAuxiliaryOperatorPlan):
        raise TypeError("plan must be a LowOrderAuxiliaryOperatorPlan.")
    if not isinstance(
        low_order_solver,
        (AbstractPreconditioner, AbstractPreconditionerBuilder),
    ):
        raise TypeError(
            "low_order_solver must be a preconditioner or preconditioner builder."
        )
    high_space = plan.interpolation.source
    diagonal = high_space.flatten(plan.multiplicity_weight)
    weighting = DiagonalLinearOperator(
        diagonal,
        space=high_space,
        operator_id=f"low-order-auxiliary-weight/{plan.plan_id}",
    )
    prolongation = weighting @ plan.anterpolation
    if (
        isinstance(plan.interpolation, AdjointLinearOperator)
        and plan.interpolation.operator is plan.anterpolation
    ):
        restriction = adjoint(prolongation)
    else:
        restriction = plan.interpolation @ weighting
    term = SubspaceCorrectionTerm(restriction, prolongation, low_order_solver)
    terms = (term,)
    if smoother is not None:
        identity = IdentityLinearOperator(high_space)
        terms = (SubspaceCorrectionTerm(identity, identity, smoother), term)
    return AdditiveSubspaceCorrectionBuilder(terms, properties=properties)


__all__ = [
    "LowOrderAuxiliaryOperatorPlan",
    "LowOrderAuxiliaryPreconditioner",
    "low_order_auxiliary_preconditioner_builder",
]
