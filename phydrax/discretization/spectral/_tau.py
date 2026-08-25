#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, PyTree

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    AbstractLinearOperator,
    ArraySpace,
    BlockLinearOperator,
    BlockSpace,
    DenseLinearOperator,
    FactorizationPolicy,
    factorize,
    LinearSolveResult,
    PreparedFactorization,
)


class GeneralizedTauPlan(StrictModule, NonTrainableState):
    """Explicit tau augmentation for one square operator and boundary constraints."""

    operator: AbstractLinearOperator
    constraint_matrix: Array
    lift_matrix: Array
    maximum_augmented_dimension: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        operator: AbstractLinearOperator,
        constraint_matrix: ArrayLike,
        lift_matrix: ArrayLike,
        /,
        *,
        maximum_augmented_dimension: int = 1024,
    ):
        if not isinstance(operator, AbstractLinearOperator):
            raise TypeError("operator must be an AbstractLinearOperator.")
        if operator.batch_shape or operator.source.size != operator.target.size:
            raise ValueError("Tau augmentation requires an unbatched square operator.")
        constraints = jnp.asarray(constraint_matrix)
        lift = jnp.asarray(lift_matrix)
        if constraints.ndim != 2 or lift.ndim != 2:
            raise ValueError("Tau constraint and lift matrices must be rank two.")
        tau_count = int(constraints.shape[0])
        if (
            tau_count <= 0
            or constraints.shape[1] != operator.source.size
            or lift.shape != (operator.target.size, tau_count)
        ):
            raise ValueError(
                "Tau matrices must have shapes (r, source.size) and (target.size, r)."
            )
        coordinate_dtype = operator.source.flatten(operator.source.zeros()).dtype
        dtype = jnp.result_type(
            constraints.dtype,
            lift.dtype,
            coordinate_dtype,
        )
        if not jnp.issubdtype(dtype, jnp.inexact):
            dtype = jnp.dtype(float)
        constraints = constraints.astype(dtype)
        lift = lift.astype(dtype)
        maximum = int(maximum_augmented_dimension)
        if maximum <= 0 or operator.source.size + tau_count > maximum:
            raise ValueError("Tau augmented system exceeds maximum_augmented_dimension.")
        self.operator = operator
        self.constraint_matrix = constraints
        self.lift_matrix = lift
        self.maximum_augmented_dimension = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "generalized-tau-plan",
                "operator": operator.operator_id,
                "constraint_shape": list(constraints.shape),
                "lift_shape": list(lift.shape),
                "maximum_augmented_dimension": maximum,
            }
        )

    @property
    def tau_count(self) -> int:
        return int(self.constraint_matrix.shape[0])

    def prepare(self, /) -> "PreparedTauSystem":
        tau_space = ArraySpace((self.tau_count,), dtype=self.lift_matrix.dtype)
        constraint_space = ArraySpace(
            (self.tau_count,),
            dtype=self.constraint_matrix.dtype,
        )
        lift = DenseLinearOperator(
            self.lift_matrix,
            source=tau_space,
            target=self.operator.target,
        )
        constraints = DenseLinearOperator(
            self.constraint_matrix,
            source=self.operator.source,
            target=constraint_space,
        )
        source = BlockSpace((self.operator.source, tau_space))
        target = BlockSpace((self.operator.target, constraint_space))
        augmented = BlockLinearOperator(
            (
                (self.operator, lift),
                (constraints, None),
            ),
            source=source,
            target=target,
            operator_id=canonical_fingerprint(
                {
                    "kind": "generalized-tau-operator",
                    "plan": self.plan_id,
                }
            ),
        )
        factorization = factorize(
            augmented,
            FactorizationPolicy("svd"),
        )
        return PreparedTauSystem(self, augmented, factorization)


class PreparedTauSystem(StrictModule, NonTrainableState):
    plan: GeneralizedTauPlan
    operator: BlockLinearOperator
    factorization: PreparedFactorization
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: GeneralizedTauPlan,
        operator: BlockLinearOperator,
        factorization: PreparedFactorization,
        /,
    ):
        self.plan = plan
        self.operator = operator
        self.factorization = factorization
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-generalized-tau-system",
                "plan": plan.plan_id,
                "operator": operator.operator_id,
                "factorization": factorization.factorization_id,
            }
        )

    def solve(
        self,
        right_hand_side: PyTree[Array],
        boundary_values: ArrayLike,
        /,
    ) -> "TauSolveResult":
        boundary = jnp.asarray(
            boundary_values,
            dtype=self.plan.constraint_matrix.dtype,
        )
        if boundary.shape != (self.plan.tau_count,):
            raise ValueError(f"boundary_values must have shape {(self.plan.tau_count,)}.")
        result = self.factorization.solve((right_hand_side, boundary))
        if not isinstance(result.value, tuple) or len(result.value) != 2:
            raise RuntimeError("Tau block solve returned an invalid block state.")
        return TauSolveResult(
            field=result.value[0],
            tau=result.value[1],
            linear_result=result,
            prepared_id=self.prepared_id,
        )


class TauSolveResult(StrictModule):
    field: PyTree[Array]
    tau: Array
    linear_result: LinearSolveResult
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        field: PyTree[Array],
        tau: ArrayLike,
        linear_result: LinearSolveResult,
        prepared_id: str,
    ):
        if not isinstance(linear_result, LinearSolveResult):
            raise TypeError("linear_result must be a LinearSolveResult.")
        self.field = field
        self.tau = jnp.asarray(tau)
        self.linear_result = linear_result
        self.prepared_id = str(prepared_id)


__all__ = [
    "GeneralizedTauPlan",
    "PreparedTauSystem",
    "TauSolveResult",
]
