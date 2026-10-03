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
from jax import Array
from jax.typing import ArrayLike

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
from ._balance_law_composition import (
    _part_callables,
    AdditiveIMEXTableau,
    ImplicitCallbacks,
)
from ._fixed_step import AbstractFixedStepMethod, FixedStepResult


class ImplicitConservationStageResult(StrictModule):
    """One diagonal implicit solve returned by an implicit-part solver.

    ``status`` is the solver's own integer code (``None`` reports zero).
    ``evidence`` is any array PyTree the solver hands to the consumer. Both
    reach ``ConservationIMEXResult`` per stage.
    """

    state: Array
    successful: Array
    iterations: Array
    residual_norm: Array
    status: Array | None = None
    evidence: Any = None


class ConservationIMEXResult(StrictModule):
    """IMEX step with aggregate and per-stage implicit-solve evidence.

    Stage arrays have one entry per tableau stage. Explicit-only stages and
    stages whose diagonal step vanishes report success with zero iterations,
    residual and status. ``stage_evidence`` holds each solver's evidence:
    ``None`` for explicit-only stages and zeros of the solver's structure when
    the diagonal step vanished.
    """

    candidate_state: Array
    accepted_state: Array
    successful: Array
    implicit_iterations: Array
    maximum_implicit_residual: Array
    method_id: str = eqx.field(static=True)
    stage_successful: Array
    stage_iterations: Array
    stage_residual_norms: Array
    stage_status: Array
    stage_evidence: tuple[Any, ...]


class ConservationIMEXMethod(StrictModule, NonTrainableState):
    """Additive IMEX step of a conservative state with per-stage solve evidence.

    ``implicit_rhs`` and ``implicit_solver`` hold one callback per implicit
    part of the tableau (a bare callable for one part). A solver may commit a
    conservative update built from fluxes at its nonlinear iterate; the stage
    rate is then that exact increment divided by the diagonal step, so
    conservation holds independently of the solve tolerance.
    """

    tableau: AdditiveIMEXTableau
    explicit_rhs: Callable = eqx.field(static=True)
    implicit_rhs: tuple[Callable, ...] = eqx.field(static=True)
    implicit_solver: tuple[Callable, ...] = eqx.field(static=True)
    validator: Callable = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        tableau: AdditiveIMEXTableau,
        explicit_rhs: Callable,
        implicit_rhs: ImplicitCallbacks,
        implicit_solver: ImplicitCallbacks,
        /,
        *,
        validator: Callable | None = None,
        method_id: str,
    ) -> None:
        if not isinstance(tableau, AdditiveIMEXTableau) or not callable(explicit_rhs):
            raise TypeError("Conservation IMEX inputs are invalid.")
        rates = _part_callables(implicit_rhs, tableau.part_count, "implicit_rhs")
        solvers = _part_callables(implicit_solver, tableau.part_count, "implicit_solver")
        validator_ = (
            (lambda state: jnp.all(jnp.isfinite(state)))
            if validator is None
            else validator
        )
        if not callable(validator_) or not str(method_id):
            raise ValueError("Conservation IMEX validator or ID is invalid.")
        self.tableau = tableau
        self.explicit_rhs = explicit_rhs
        self.implicit_rhs = rates
        self.implicit_solver = solvers
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
        tableau = self.tableau
        time_ = jnp.asarray(time)
        value = jnp.asarray(state)
        step = jnp.asarray(step_size)

        # Infer the numerical state dtype without evaluating an extra RHS or
        # solve. Promote from the RHS before tracing a dtype-sensitive solver.
        structures = eqx.filter_eval_shape(
            lambda candidate: (
                jnp.asarray(self.explicit_rhs(time_, candidate, args)),
                *(
                    jnp.asarray(rate(time_, candidate, args))
                    for rate in self.implicit_rhs
                ),
            ),
            value,
        )
        value = value.astype(
            jnp.result_type(
                value,
                step,
                tableau.weights,
                *(structure.dtype for structure in structures),
            )
        )
        # One trace per part solver yields the solved dtype and the evidence
        # structure that a stage with a vanishing diagonal step zero-fills.
        part_structures = eqx.filter_eval_shape(
            lambda candidate: tuple(
                (jnp.asarray(result.state), result.evidence)
                for result in (
                    _checked(solver(candidate, time_, step, args))
                    for solver in self.implicit_solver
                )
            ),
            value,
        )
        value = value.astype(
            jnp.result_type(value, *(state.dtype for state, _ in part_structures))
        )
        explicit_rates: list[Array | None] = []
        implicit_rates: list[Array] = []
        stage_values: list[Array] = []
        records: list[ImplicitConservationStageResult] = []
        for stage in range(tableau.stage_count):
            provisional = tableau.provisional(
                stage, value, step, explicit_rates, implicit_rates, stage_values
            )
            stage_time = time_ + tableau.nodes[stage] * step
            part = tableau.implicit_parts[stage]
            if part is None:
                solved = provisional
                rate = jnp.zeros_like(provisional)
                record = _passive_stage(provisional, None)
            else:
                solved, rate, record = self._implicit_stage(
                    part,
                    step * tableau.implicit_matrix[stage, stage],
                    provisional,
                    stage_time,
                    args,
                    part_structures[part][1],
                )
            stage_values.append(solved)
            implicit_rates.append(rate)
            records.append(record)
            explicit_rates.append(
                self.explicit_rhs(stage_time, solved, args)
                if tableau.explicit_rate_used(stage)
                else None
            )
        candidate = tableau.combine(
            value, step, explicit_rates, implicit_rates, stage_values
        )
        stage_successful = jnp.stack([record.successful for record in records])
        stage_iterations = jnp.stack([record.iterations for record in records])
        stage_residuals = jnp.stack([record.residual_norm for record in records])
        successful = (
            jnp.all(jnp.isfinite(value))
            & jnp.isfinite(time_)
            & jnp.isfinite(step)
            & jnp.all(stage_successful)
            & jnp.all(jnp.isfinite(candidate))
            & self.validator(candidate)
        )
        return ConservationIMEXResult(
            candidate,
            jnp.where(successful, candidate, value),
            successful,
            jnp.sum(stage_iterations, dtype=jnp.int32),
            jnp.max(stage_residuals),
            self.method_id,
            stage_successful,
            stage_iterations,
            stage_residuals,
            jnp.stack([jnp.asarray(record.status) for record in records]),
            tuple(record.evidence for record in records),
        )

    def _implicit_stage(
        self,
        part: int,
        coefficient: Array,
        provisional: Array,
        stage_time: Array,
        args: Any,
        evidence_structure: Any,
        /,
    ) -> tuple[Array, Array, ImplicitConservationStageResult]:
        solver = self.implicit_solver[part]
        result = jax.lax.cond(
            coefficient != 0.0,
            lambda value: _normalized(
                _checked(solver(value, stage_time, coefficient, args)), value
            ),
            lambda value: _passive_stage(value, evidence_structure),
            provisional,
        )
        stage_successful = (
            result.successful
            & jnp.all(jnp.isfinite(result.state))
            & jnp.isfinite(result.residual_norm)
        )
        solved = jnp.where(stage_successful, result.state, provisional)
        # Mask the denominator before division: an inactive 0/0 branch
        # otherwise poisons reverse-mode derivatives through jnp.where.
        safe_coefficient = jnp.where(coefficient != 0.0, coefficient, 1.0)
        rate = jnp.where(
            coefficient != 0.0,
            (solved - provisional) / safe_coefficient,
            self.implicit_rhs[part](stage_time, solved, args),
        )
        record = ImplicitConservationStageResult(
            solved,
            stage_successful,
            result.iterations,
            result.residual_norm,
            result.status,
            result.evidence,
        )
        return solved, rate, record


class ConservationIMEXFixedStepMethod(AbstractFixedStepMethod, NonTrainableState):
    """One conservative IMEX step published to native fixed-step rollouts.

    Candidate, accepted state and success are the IMEX method's own (including
    its validator); ``evidence`` retains per-stage solve success, iterations,
    residual norms and status with an outcome-independent structure. Native
    rollouts hold the state after the first refusal; no retry, clipping or
    stabilization is added here.
    """

    method: ConservationIMEXMethod
    method_id: str = eqx.field(static=True)

    def __init__(self, method: ConservationIMEXMethod, /) -> None:
        if not isinstance(method, ConservationIMEXMethod):
            raise TypeError("method must be a ConservationIMEXMethod.")
        self.method = method
        self.method_id = canonical_fingerprint(
            {"kind": "conservation-imex-fixed-step", "method": method.method_id}
        )

    def step(
        self,
        step_index: Array,
        time: Array,
        state: Array,
        step_size: Array,
        args: Any,
        /,
    ) -> FixedStepResult:
        del step_index
        result = self.method.step(time, state, step_size, args)
        candidate = result.candidate_state
        return FixedStepResult(
            candidate,
            result.accepted_state,
            result.successful,
            jnp.asarray(result.maximum_implicit_residual, dtype=candidate.dtype),
            result.implicit_iterations,
            jnp.asarray(self.method.tableau.stage_count, dtype=jnp.int32),
            jnp.asarray(False),
            jnp.zeros((), dtype=candidate.real.dtype),
            evidence={
                "stage_successful": result.stage_successful,
                "stage_iterations": result.stage_iterations,
                "stage_residual_norms": result.stage_residual_norms,
                "stage_status": result.stage_status,
            },
        )


def _checked(result: Any, /) -> ImplicitConservationStageResult:
    if not isinstance(result, ImplicitConservationStageResult):
        raise TypeError("Implicit conservation solver must return stage result.")
    return result


def _normalized(
    result: ImplicitConservationStageResult, provisional: Array, /
) -> ImplicitConservationStageResult:
    return ImplicitConservationStageResult(
        jnp.asarray(result.state, dtype=provisional.dtype),
        jnp.asarray(result.successful, dtype=jnp.bool_),
        jnp.asarray(result.iterations, dtype=jnp.int32),
        jnp.asarray(jnp.abs(result.residual_norm), dtype=provisional.real.dtype),
        jnp.asarray(0 if result.status is None else result.status, dtype=jnp.int32),
        result.evidence,
    )


def _passive_stage(
    provisional: Array, evidence: Any, /
) -> ImplicitConservationStageResult:
    """Return the record of a stage without a solve; evidence is zero-filled."""
    return ImplicitConservationStageResult(
        provisional,
        jnp.asarray(True),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.zeros((), dtype=provisional.real.dtype),
        jnp.asarray(0, dtype=jnp.int32),
        jax.tree.map(lambda leaf: jnp.zeros(leaf.shape, leaf.dtype), evidence),
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
    ) -> None:
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
        component_count = int(np.prod(value.shape[1:], dtype=np.int64))
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
    "ConservationIMEXFixedStepMethod",
    "ConservationIMEXMethod",
    "ConservationIMEXResult",
    "ElementBlockPreconditioner",
    "ImplicitConservationStageResult",
    "prepare_element_block_preconditioner",
]
