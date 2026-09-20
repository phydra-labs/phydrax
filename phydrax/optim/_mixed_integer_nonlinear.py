#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._programming._mixed_integer import _indices


ConvexConstraintKind: TypeAlias = Literal[
    "convex-upper",
    "concave-lower",
    "affine-equality",
]


class ConvexConstraintEvidence(StrictModule, NonTrainableState):
    kind: ConvexConstraintKind = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(self, kind: ConvexConstraintKind, evidence_id: str, /):
        if kind not in ("convex-upper", "concave-lower", "affine-equality"):
            raise ValueError("Unknown convex constraint evidence kind.")
        identifier = str(evidence_id)
        if not identifier:
            raise ValueError("evidence_id must be nonempty.")
        self.kind = kind
        self.evidence_id = identifier


class ConvexMINLPEvaluation(StrictModule, NonTrainableState):
    primal: Array
    objective: Array
    gradient: Array
    constraints: Array
    jacobian: Array
    finite: Array


class ConvexMINLPCandidateAudit(StrictModule, NonTrainableState):
    primal: Array
    objective: Array
    constraint_violation: Array
    bound_violation: Array
    integrality_violation: Array
    finite: Array
    valid: Array
    program_id: str = eqx.field(static=True)
    structure_id: str = eqx.field(static=True)


class ConvexMixedIntegerNonlinearProgram(StrictModule):
    """Compact differentiable convex MINLP with proof-carrying row orientation."""

    objective: Callable[[Array, Any], Array] = eqx.field(static=True)
    constraints: Callable[[Array, Any], Array] = eqx.field(static=True)
    args: Any
    variable_lower: Array
    variable_upper: Array
    constraint_lower: Array
    constraint_upper: Array
    constraint_evidence: tuple[ConvexConstraintEvidence, ...]
    integer_indices: tuple[int, ...] = eqx.field(static=True)
    binary_indices: tuple[int, ...] = eqx.field(static=True)
    discrete_indices: tuple[int, ...] = eqx.field(static=True)
    objective_evidence_id: str = eqx.field(static=True)
    num_variables: int = eqx.field(static=True)
    num_constraints: int = eqx.field(static=True)
    program_id: str = eqx.field(static=True)
    structure_id: str = eqx.field(static=True)

    def __init__(
        self,
        objective: Callable[[Array, Any], Array],
        constraints: Callable[[Array, Any], Array],
        variable_lower: ArrayLike,
        variable_upper: ArrayLike,
        constraint_lower: ArrayLike,
        constraint_upper: ArrayLike,
        /,
        *,
        constraint_evidence: Sequence[ConvexConstraintEvidence],
        integer_indices: Sequence[int] = (),
        binary_indices: Sequence[int] = (),
        args: Any = None,
        objective_evidence_id: str,
        program_id: str = "convex-mixed-integer-nonlinear-program",
    ):
        if not callable(objective) or not callable(constraints):
            raise TypeError("objective and constraints must be callable.")
        lower = jnp.asarray(variable_lower)
        upper = jnp.asarray(variable_upper)
        constraint_lower_ = jnp.asarray(constraint_lower)
        constraint_upper_ = jnp.asarray(constraint_upper)
        if lower.ndim != 1 or upper.shape != lower.shape or lower.size < 1:
            raise ValueError("Variable bounds must be equal nonempty vectors.")
        if (
            constraint_lower_.ndim != 1
            or constraint_upper_.shape != constraint_lower_.shape
        ):
            raise ValueError("Constraint bounds must be equal vectors.")
        dtype = jnp.result_type(
            lower, upper, constraint_lower_, constraint_upper_, jnp.float32
        )
        if not jnp.issubdtype(dtype, jnp.floating):
            raise TypeError("Convex MINLP bounds must be real floating arrays.")
        lower = lower.astype(dtype)
        upper = upper.astype(dtype)
        constraint_lower_ = constraint_lower_.astype(dtype)
        constraint_upper_ = constraint_upper_.astype(dtype)
        lower_host, upper_host = np.asarray(lower), np.asarray(upper)
        if not np.all(np.isfinite(lower_host)) or not np.all(np.isfinite(upper_host)):
            raise ValueError("Convex MINLP variable bounds must be finite.")
        if np.any(lower_host > upper_host):
            raise ValueError("Variable lower bounds cannot exceed upper bounds.")
        constraint_lower_host = np.asarray(constraint_lower_)
        constraint_upper_host = np.asarray(constraint_upper_)
        if np.any(np.isnan(constraint_lower_host)) or np.any(
            np.isnan(constraint_upper_host)
        ):
            raise ValueError("Constraint bounds cannot contain NaN.")
        if np.any(constraint_lower_host > constraint_upper_host):
            raise ValueError("Constraint lower bounds cannot exceed upper bounds.")
        evidence = tuple(constraint_evidence)
        constraints_count = constraint_lower_.size
        if len(evidence) != constraints_count or any(
            not isinstance(value, ConvexConstraintEvidence) for value in evidence
        ):
            raise TypeError("constraint_evidence must identify every constraint row.")
        for index, value in enumerate(evidence):
            lo = constraint_lower_host[index]
            hi = constraint_upper_host[index]
            equality = np.isfinite(lo) and np.isfinite(hi) and lo == hi
            upper_only = not np.isfinite(lo) and np.isfinite(hi)
            lower_only = np.isfinite(lo) and not np.isfinite(hi)
            expected = (
                "affine-equality"
                if equality
                else "convex-upper"
                if upper_only
                else "concave-lower"
                if lower_only
                else None
            )
            if value.kind != expected:
                raise ValueError(
                    "Constraint bounds and curvature orientation are inconsistent."
                )
        variables = lower.size
        integer = _indices(integer_indices, variables, "integer")
        binary = _indices(binary_indices, variables, "binary")
        if set(integer) & set(binary):
            raise ValueError("integer_indices and binary_indices must be disjoint.")
        discrete = tuple(sorted((*integer, *binary)))
        if not discrete:
            raise ValueError("Convex MINLP requires at least one discrete coordinate.")
        discrete_array = np.asarray(discrete, dtype=np.int64)
        if np.any(
            lower_host[discrete_array] != np.ceil(lower_host[discrete_array])
        ) or np.any(upper_host[discrete_array] != np.floor(upper_host[discrete_array])):
            raise ValueError("Discrete variable bounds must be integral.")
        if binary:
            binary_array = np.asarray(binary, dtype=np.int64)
            if np.any(lower_host[binary_array] < 0.0) or np.any(
                upper_host[binary_array] > 1.0
            ):
                raise ValueError("Binary variable bounds must lie inside [0, 1].")
        objective_evidence = str(objective_evidence_id)
        identifier = str(program_id)
        if not objective_evidence or not identifier:
            raise ValueError("Objective evidence and program_id must be nonempty.")
        sample = 0.5 * (lower + upper)
        objective_sample = jnp.asarray(objective(sample, args))
        constraints_sample = jnp.asarray(constraints(sample, args))
        if objective_sample.shape != () or not jnp.issubdtype(
            objective_sample.dtype, jnp.floating
        ):
            raise TypeError("objective must return one real scalar.")
        if constraints_sample.shape != (constraints_count,) or not jnp.issubdtype(
            constraints_sample.dtype, jnp.floating
        ):
            raise TypeError(f"constraints must return shape ({constraints_count},).")
        if not bool(
            np.asarray(
                jnp.isfinite(objective_sample) & jnp.all(jnp.isfinite(constraints_sample))
            )
        ):
            raise ValueError(
                "Objective and constraints must be finite at the box midpoint."
            )
        self.objective = objective
        self.constraints = constraints
        self.args = args
        self.variable_lower = lower
        self.variable_upper = upper
        self.constraint_lower = constraint_lower_
        self.constraint_upper = constraint_upper_
        self.constraint_evidence = evidence
        self.integer_indices = integer
        self.binary_indices = binary
        self.discrete_indices = discrete
        self.objective_evidence_id = objective_evidence
        self.num_variables = variables
        self.num_constraints = constraints_count
        self.program_id = identifier
        self.structure_id = canonical_fingerprint(
            {
                "kind": "convex-minlp",
                "program_id": identifier,
                "variables": variables,
                "constraints": constraints_count,
                "integer": list(integer),
                "binary": list(binary),
                "objective_evidence": objective_evidence,
                "constraint_evidence": [
                    {"kind": value.kind, "id": value.evidence_id} for value in evidence
                ],
                "bounds": array_tree_fingerprint(
                    (lower, upper, constraint_lower_, constraint_upper_)
                ),
                "args": array_tree_fingerprint(args),
            }
        )

    def evaluate(self, primal: ArrayLike, /) -> ConvexMINLPEvaluation:
        value = jnp.asarray(primal, dtype=self.variable_lower.dtype)
        if value.shape != (self.num_variables,):
            raise ValueError(f"primal must have shape ({self.num_variables},).")

        def scalar(point):
            return jnp.asarray(self.objective(point, self.args))

        objective, gradient = jax.value_and_grad(scalar)(value)
        constraints = jnp.asarray(self.constraints(value, self.args))
        jacobian = jax.jacrev(lambda point: self.constraints(point, self.args))(value)
        finite = (
            jnp.isfinite(objective)
            & jnp.all(jnp.isfinite(gradient))
            & jnp.all(jnp.isfinite(constraints))
            & jnp.all(jnp.isfinite(jacobian))
        )
        return ConvexMINLPEvaluation(
            value,
            objective,
            gradient,
            constraints,
            jacobian,
            finite,
        )

    def audit(
        self,
        primal: ArrayLike,
        /,
        *,
        feasibility_tolerance: float,
        integrality_tolerance: float,
    ) -> ConvexMINLPCandidateAudit:
        evaluation = self.evaluate(primal)
        value = evaluation.primal
        bound_violation = jnp.max(
            jnp.maximum(
                jnp.maximum(self.variable_lower - value, value - self.variable_upper),
                0.0,
            ),
            initial=0.0,
        )
        constraint_violation = jnp.max(
            jnp.maximum(
                jnp.maximum(
                    self.constraint_lower - evaluation.constraints,
                    evaluation.constraints - self.constraint_upper,
                ),
                0.0,
            ),
            initial=0.0,
        )
        discrete = value[jnp.asarray(self.discrete_indices, dtype=jnp.int32)]
        integrality = jnp.max(
            jnp.abs(discrete - jnp.rint(discrete)),
            initial=0.0,
        )
        finite = evaluation.finite & jnp.all(jnp.isfinite(value))
        valid = (
            finite
            & (bound_violation <= float(feasibility_tolerance))
            & (constraint_violation <= float(feasibility_tolerance))
            & (integrality <= float(integrality_tolerance))
        )
        return ConvexMINLPCandidateAudit(
            value,
            evaluation.objective,
            constraint_violation,
            bound_violation,
            integrality,
            finite,
            valid,
            self.program_id,
            self.structure_id,
        )


__all__ = [
    "ConvexConstraintEvidence",
    "ConvexConstraintKind",
    "ConvexMINLPCandidateAudit",
    "ConvexMINLPEvaluation",
    "ConvexMixedIntegerNonlinearProgram",
]
