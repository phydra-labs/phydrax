#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import (
    DenseLinearOperator,
    FactorizationPolicy,
    factorize,
    OperatorProperties,
    PreparedFactorization,
)
from ._balance_law_composition import AdditiveIMEXTableau


class ImplicitConservationStageResult(StrictModule):
    state: Array
    successful: Array
    iterations: Array
    residual_norm: Array


class ConservationIMEXResult(StrictModule):
    candidate_state: Array
    accepted_state: Array
    successful: Array
    implicit_iterations: Array
    maximum_implicit_residual: Array
    method_id: str = eqx.field(static=True)


class ConservationIMEXMethod(StrictModule, NonTrainableState):
    tableau: AdditiveIMEXTableau
    explicit_rhs: Callable = eqx.field(static=True)
    implicit_rhs: Callable = eqx.field(static=True)
    implicit_solver: Callable = eqx.field(static=True)
    validator: Callable = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        tableau: AdditiveIMEXTableau,
        explicit_rhs: Callable,
        implicit_rhs: Callable,
        implicit_solver: Callable,
        /,
        *,
        validator: Callable | None = None,
        method_id: str,
    ):
        if (
            not isinstance(tableau, AdditiveIMEXTableau)
            or not callable(explicit_rhs)
            or not callable(implicit_rhs)
            or not callable(implicit_solver)
        ):
            raise TypeError("Conservation IMEX inputs are invalid.")
        validator_ = lambda state: (
            jnp.all(jnp.isfinite(state)) if validator is None else validator
        )
        if not callable(validator_) or not str(method_id):
            raise ValueError("Conservation IMEX validator or ID is invalid.")
        self.tableau = tableau
        self.explicit_rhs = explicit_rhs
        self.implicit_rhs = implicit_rhs
        self.implicit_solver = implicit_solver
        self.validator = validator_
        self.method_id = canonical_fingerprint(
            {
                "kind": "conservation-imex-method",
                "tableau": tableau.tableau_id,
                "method": str(method_id),
            }
        )

    def step(
        self,
        time: Array,
        state: Array,
        step_size: Array,
        args: Any = None,
        /,
    ) -> ConservationIMEXResult:
        time_ = jnp.asarray(time)
        value = jnp.asarray(state)
        step = jnp.asarray(step_size)
        explicit_stages = []
        implicit_stages = []
        successful = jnp.asarray(True)
        iterations = jnp.asarray(0, dtype=jnp.int32)
        maximum_residual = jnp.zeros((), dtype=value.dtype)
        for stage in range(self.tableau.stage_count):
            provisional = value
            for previous in range(stage):
                provisional = provisional + step * (
                    self.tableau.explicit_matrix[stage, previous]
                    * explicit_stages[previous]
                    + self.tableau.implicit_matrix[stage, previous]
                    * implicit_stages[previous]
                )
            stage_time = time_ + self.tableau.nodes[stage] * step
            diagonal = self.tableau.implicit_matrix[stage, stage]
            stage_result = self.implicit_solver(
                provisional,
                stage_time,
                step * diagonal,
                args,
            )
            if not isinstance(stage_result, ImplicitConservationStageResult):
                raise TypeError("Implicit conservation solver must return stage result.")
            solved = jnp.where(stage_result.successful, stage_result.state, value)
            explicit_stages.append(self.explicit_rhs(stage_time, solved, args))
            implicit_stages.append(
                jnp.where(
                    diagonal != 0.0,
                    (solved - provisional) / (step * diagonal),
                    self.implicit_rhs(stage_time, solved, args),
                )
            )
            successful = successful & stage_result.successful
            iterations = iterations + stage_result.iterations
            maximum_residual = jnp.maximum(maximum_residual, stage_result.residual_norm)
        candidate = value
        for stage in range(self.tableau.stage_count):
            candidate = candidate + step * self.tableau.weights[stage] * (
                explicit_stages[stage] + implicit_stages[stage]
            )
        successful = successful & self.validator(candidate)
        return ConservationIMEXResult(
            candidate,
            jnp.where(successful, candidate, value),
            successful,
            iterations,
            maximum_residual,
            self.method_id,
        )


class ElementBlockPreconditioner(StrictModule):
    routes: tuple[Array, ...]
    factorizations: tuple[PreparedFactorization, ...]
    preconditioner_id: str = eqx.field(static=True)

    def __init__(
        self,
        routes: Sequence[ArrayLike],
        blocks: Sequence[ArrayLike],
        /,
        *,
        preconditioner_id: str,
    ):
        routes_ = tuple(jnp.asarray(value, dtype=jnp.int32) for value in routes)
        blocks_ = tuple(jnp.asarray(value) for value in blocks)
        if (
            not routes_
            or len(routes_) != len(blocks_)
            or any(route.ndim != 1 for route in routes_)
            or any(
                block.ndim != 2 or block.shape[0] != block.shape[1] for block in blocks_
            )
        ):
            raise ValueError("Element block preconditioner shapes are invalid.")
        properties = OperatorProperties()
        factors = tuple(
            factorize(
                DenseLinearOperator(
                    block,
                    properties=properties,
                    operator_id=canonical_fingerprint(
                        {
                            "kind": "element-block-preconditioner",
                            "index": index,
                            "shape": block.shape,
                        }
                    ),
                ),
                FactorizationPolicy("lu"),
            )
            for index, block in enumerate(blocks_)
        )
        self.routes = routes_
        self.factorizations = factors
        self.preconditioner_id = canonical_fingerprint(
            {
                "kind": "element-block-preconditioner-plan",
                "name": str(preconditioner_id),
                "route_count": len(routes_),
            }
        )

    def apply(self, residual: ArrayLike, /) -> Array:
        value = jnp.asarray(residual)
        result = jnp.zeros_like(value)
        for route, factor in zip(self.routes, self.factorizations, strict=True):
            local = value[route]
            flat = local.reshape((-1,))
            solved = factor.solve(flat)
            local_result = eqx.error_if(
                solved.value,
                ~solved.successful,
                "Element block preconditioner solve failed.",
            ).reshape(local.shape)
            result = result.at[route].set(local_result)
        return result


def prepare_element_block_preconditioner(
    state: ArrayLike,
    routes: Sequence[ArrayLike],
    implicit_rate: Callable,
    /,
    *,
    time: ArrayLike,
    step_coefficient: ArrayLike,
    args: Any = None,
) -> ElementBlockPreconditioner:
    value = jnp.asarray(state)
    coefficient = jnp.asarray(step_coefficient)
    jacobian = jax.jacfwd(
        lambda candidate: candidate - coefficient * implicit_rate(time, candidate, args)
    )(value)
    blocks = []
    route_values = []
    for route in routes:
        indices = jnp.asarray(route, dtype=jnp.int32)
        component_count = int(np.prod(value.shape[1:], dtype=int))
        flattened_indices = (
            indices[:, None] * component_count + jnp.arange(component_count)[None, :]
        ).reshape((-1,))
        matrix = jacobian.reshape((value.size, value.size))
        blocks.append(matrix[flattened_indices[:, None], flattened_indices[None, :]])
        route_values.append(indices)
    return ElementBlockPreconditioner(
        tuple(route_values),
        tuple(blocks),
        preconditioner_id=canonical_fingerprint(
            {
                "kind": "prepared-element-block-preconditioner",
                "route_count": len(route_values),
                "step_coefficient": float(np.asarray(coefficient)),
            }
        ),
    )


__all__ = [
    "ConservationIMEXMethod",
    "ConservationIMEXResult",
    "ElementBlockPreconditioner",
    "ImplicitConservationStageResult",
    "prepare_element_block_preconditioner",
]
