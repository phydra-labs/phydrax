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

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from ..sparse import SparseCoordinateOperator, SparseDerivativePlan
from ._iterative import ConstrainedOptimalityCertificate, MinimizationResult


_UNSET = object()


def _identifier(value: str, owner: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} must be a non-empty string.")
    return value


def _real_vector(value: ArrayLike, size: int | None, owner: str, /) -> Array:
    array = jnp.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{owner} must be rank one.")
    if size is not None and array.shape != (size,):
        raise ValueError(f"{owner} must have shape {(size,)}; got {array.shape}.")
    if jnp.issubdtype(array.dtype, jnp.complexfloating):
        raise TypeError(f"{owner} must be real-valued.")
    return array if jnp.issubdtype(array.dtype, jnp.inexact) else array.astype(float)


def _maximum(values: Array, /) -> Array:
    return jnp.max(values, initial=jnp.asarray(0.0, dtype=values.dtype))


def _numeric_fingerprint(values: Any, /) -> str:
    leaves = tuple(
        leaf
        for leaf in jax.tree_util.tree_leaves(values)
        if isinstance(leaf, (jax.Array, np.ndarray, jax.core.Tracer))
    )
    if any(isinstance(leaf, jax.core.Tracer) for leaf in leaves):
        return "traced-structured-numerics"
    return array_tree_fingerprint(leaves)


_STRUCTURED_WORK_FIELDS = (
    "objective_evaluations",
    "constraint_evaluations",
    "gradient_evaluations",
    "jacobian_evaluations",
    "hessian_evaluations",
    "kkt_assemblies",
    "symbolic_analyses",
    "factorizations",
    "numeric_refactorizations",
    "right_hand_side_solves",
    "refinement_steps",
    "backtracking_evaluations",
    "second_order_corrections",
    "restoration_evaluations",
    "restoration_solves",
    "certificate_evaluations",
    "provider_rebuilds",
)


def _work_count(value: Any, name: str, /) -> Array:
    count = jnp.asarray(value, dtype=jnp.int32)
    if count.shape != ():
        raise ValueError(f"{name} must be scalar.")
    return eqx.error_if(count, count < 0, f"{name} must be non-negative.")


class StructuredOptimizationWork(StrictModule):
    """Exact associative work evidence for one structured optimization solve."""

    objective_evaluations: Array
    constraint_evaluations: Array
    gradient_evaluations: Array
    jacobian_evaluations: Array
    hessian_evaluations: Array
    kkt_assemblies: Array
    symbolic_analyses: Array
    factorizations: Array
    numeric_refactorizations: Array
    right_hand_side_solves: Array
    refinement_steps: Array
    backtracking_evaluations: Array
    second_order_corrections: Array
    restoration_evaluations: Array
    restoration_solves: Array
    certificate_evaluations: Array
    provider_rebuilds: Array
    complete: Array

    def __init__(
        self,
        *,
        objective_evaluations: Any = 0,
        constraint_evaluations: Any = 0,
        gradient_evaluations: Any = 0,
        jacobian_evaluations: Any = 0,
        hessian_evaluations: Any = 0,
        kkt_assemblies: Any = 0,
        symbolic_analyses: Any = 0,
        factorizations: Any = 0,
        numeric_refactorizations: Any = 0,
        right_hand_side_solves: Any = 0,
        refinement_steps: Any = 0,
        backtracking_evaluations: Any = 0,
        second_order_corrections: Any = 0,
        restoration_evaluations: Any = 0,
        restoration_solves: Any = 0,
        certificate_evaluations: Any = 0,
        provider_rebuilds: Any = 0,
        complete: Any = True,
    ):
        values = locals()
        for name in _STRUCTURED_WORK_FIELDS:
            setattr(self, name, _work_count(values[name], name))
        complete_ = jnp.asarray(complete, dtype=bool)
        if complete_.shape != ():
            raise ValueError("complete must be scalar.")
        self.complete = complete_

    @classmethod
    def zero(cls, /, *, complete: Any = True) -> StructuredOptimizationWork:
        return cls(complete=complete)

    def __add__(self, other: object, /) -> StructuredOptimizationWork:
        if not isinstance(other, StructuredOptimizationWork):
            return NotImplemented
        values = {
            name: getattr(self, name) + getattr(other, name)
            for name in _STRUCTURED_WORK_FIELDS
        }
        return StructuredOptimizationWork(
            **values,
            complete=self.complete & other.complete,
        )


class StructuredNonlinearWarmStart(StrictModule):
    """Primal and bound-form dual values for one structured nonlinear program."""

    primal: Array
    constraint_multipliers: Array
    lower_bound_multipliers: Array
    upper_bound_multipliers: Array
    numeric_version: Array
    structure_id: str = eqx.field(static=True)
    source_result_id: str | None = eqx.field(static=True)
    source_program_id: str | None = eqx.field(static=True)
    source_backend: str | None = eqx.field(static=True)
    warm_start_id: str = eqx.field(static=True)

    def __init__(
        self,
        primal: ArrayLike,
        constraint_multipliers: ArrayLike,
        lower_bound_multipliers: ArrayLike,
        upper_bound_multipliers: ArrayLike,
        /,
        *,
        structure_id: str,
        numeric_version: Any = 0,
        source_result_id: str | None = None,
        source_program_id: str | None = None,
        source_backend: str | None = None,
        warm_start_id: str | None = None,
    ):
        primal_ = _real_vector(primal, None, "warm-start primal")
        constraints = _real_vector(
            constraint_multipliers, None, "warm-start constraint multipliers"
        )
        lower = _real_vector(
            lower_bound_multipliers, int(primal_.size), "warm-start lower multipliers"
        )
        upper = _real_vector(
            upper_bound_multipliers, int(primal_.size), "warm-start upper multipliers"
        )
        primal_ = eqx.error_if(
            primal_,
            jnp.any(~jnp.isfinite(primal_)),
            "warm-start primal must be finite.",
        )
        constraints = eqx.error_if(
            constraints,
            jnp.any(~jnp.isfinite(constraints)),
            "warm-start constraint multipliers must be finite.",
        )
        lower = eqx.error_if(
            lower,
            jnp.any(~jnp.isfinite(lower)) | jnp.any(lower < 0.0),
            "warm-start lower multipliers must be finite and non-negative.",
        )
        upper = eqx.error_if(
            upper,
            jnp.any(~jnp.isfinite(upper)) | jnp.any(upper < 0.0),
            "warm-start upper multipliers must be finite and non-negative.",
        )
        version = jnp.asarray(numeric_version, dtype=jnp.int32)
        if version.shape != ():
            raise ValueError("numeric_version must be scalar.")
        version = eqx.error_if(
            version,
            version < 0,
            "numeric_version must be non-negative.",
        )
        structure = _identifier(structure_id, "structure_id")
        self.primal = primal_
        self.constraint_multipliers = constraints
        self.lower_bound_multipliers = lower
        self.upper_bound_multipliers = upper
        self.numeric_version = version
        self.structure_id = structure
        self.source_result_id = (
            None
            if source_result_id is None
            else _identifier(source_result_id, "source_result_id")
        )
        self.source_program_id = (
            None
            if source_program_id is None
            else _identifier(source_program_id, "source_program_id")
        )
        self.source_backend = (
            None
            if source_backend is None
            else _identifier(source_backend, "source_backend")
        )
        self.warm_start_id = (
            canonical_fingerprint(
                {
                    "kind": "structured-nonlinear-warm-start",
                    "structure": structure,
                    "numeric_version": (
                        "traced"
                        if isinstance(version, jax.core.Tracer)
                        else int(np.asarray(version))
                    ),
                    "source_result": self.source_result_id,
                    "source_program": self.source_program_id,
                    "source_backend": self.source_backend,
                    "primal_size": int(primal_.size),
                    "constraint_size": int(constraints.size),
                    "values": _numeric_fingerprint((primal_, constraints, lower, upper)),
                }
            )
            if warm_start_id is None
            else _identifier(warm_start_id, "warm_start_id")
        )


class StructuredNonlinearEvaluation(StrictModule):
    """Objective, constraints, gradient, and sparse Jacobian at one point."""

    coordinates: Array
    objective: Array
    gradient: Array
    constraints: Array
    jacobian: SparseCoordinateOperator
    finite: Array


class StructuredNonlinearProgram(StrictModule):
    """Bound-form nonlinear program with reusable exact sparse derivatives."""

    objective: Callable[[Array, Any], ArrayLike]
    constraints: Callable[[Array, Any], ArrayLike]
    jacobian_plan: SparseDerivativePlan
    hessian_plan: SparseDerivativePlan | None
    variable_lower: Array
    variable_upper: Array
    constraint_lower: Array
    constraint_upper: Array
    equality_indices: Array
    lower_indices: Array
    upper_indices: Array
    constraint_sources: tuple[str, ...] = eqx.field(static=True)
    num_variables: int = eqx.field(static=True)
    num_constraints: int = eqx.field(static=True)
    program_id: str = eqx.field(static=True)
    structure_id: str = eqx.field(static=True)

    def __init__(
        self,
        objective: Callable[[Array, Any], ArrayLike],
        constraints: Callable[[Array, Any], ArrayLike],
        jacobian_plan: SparseDerivativePlan,
        /,
        *,
        variable_lower: ArrayLike,
        variable_upper: ArrayLike,
        constraint_lower: ArrayLike,
        constraint_upper: ArrayLike,
        constraint_sources: Sequence[str],
        hessian_plan: SparseDerivativePlan | None = None,
        program_id: str,
        structure_id: str,
    ):
        if not callable(objective) or not callable(constraints):
            raise TypeError(
                "Structured nonlinear objective and constraints must be callable."
            )
        if not isinstance(jacobian_plan, SparseDerivativePlan):
            raise TypeError("jacobian_plan must be a SparseDerivativePlan.")
        if hessian_plan is not None and not isinstance(
            hessian_plan, SparseDerivativePlan
        ):
            raise TypeError("hessian_plan must be a SparseDerivativePlan or None.")
        variables = jacobian_plan.pattern.source_size
        constraint_count = jacobian_plan.pattern.target_size
        lower_x = _real_vector(variable_lower, variables, "variable_lower")
        upper_x = _real_vector(variable_upper, variables, "variable_upper")
        lower_c = _real_vector(constraint_lower, constraint_count, "constraint_lower")
        upper_c = _real_vector(constraint_upper, constraint_count, "constraint_upper")
        if bool(np.any(np.asarray(lower_x) > np.asarray(upper_x))):
            raise ValueError("Variable lower bounds must not exceed upper bounds.")
        if bool(np.any(np.asarray(lower_c) > np.asarray(upper_c))):
            raise ValueError("Constraint lower bounds must not exceed upper bounds.")
        if hessian_plan is not None and hessian_plan.pattern.shape != (
            variables,
            variables,
        ):
            raise ValueError("hessian_plan must have one square variable-space pattern.")
        sources = tuple(str(source) for source in constraint_sources)
        if len(sources) != constraint_count or any(not source for source in sources):
            raise ValueError("constraint_sources must identify every constraint scalar.")
        lower_host = np.asarray(lower_c)
        upper_host = np.asarray(upper_c)
        equality = np.flatnonzero(
            np.isfinite(lower_host) & np.isfinite(upper_host) & (lower_host == upper_host)
        )
        lower = np.flatnonzero(np.isfinite(lower_host) & (lower_host != upper_host))
        upper = np.flatnonzero(np.isfinite(upper_host) & (lower_host != upper_host))
        self.objective = objective
        self.constraints = constraints
        self.jacobian_plan = jacobian_plan
        self.hessian_plan = hessian_plan
        self.variable_lower = lower_x
        self.variable_upper = upper_x
        self.constraint_lower = lower_c
        self.constraint_upper = upper_c
        self.equality_indices = jnp.asarray(equality, dtype=jnp.int32)
        self.lower_indices = jnp.asarray(lower, dtype=jnp.int32)
        self.upper_indices = jnp.asarray(upper, dtype=jnp.int32)
        self.constraint_sources = sources
        self.num_variables = variables
        self.num_constraints = constraint_count
        self.program_id = _identifier(program_id, "program_id")
        self.structure_id = _identifier(structure_id, "structure_id")

    def validate_coordinates(self, coordinates: ArrayLike, /) -> Array:
        return _real_vector(coordinates, self.num_variables, "coordinates")

    def evaluate(self, coordinates: ArrayLike, args: Any = None, /):
        point = self.validate_coordinates(coordinates)

        def scalar(value):
            output = jnp.asarray(self.objective(value, args))
            if output.shape != () or not jnp.issubdtype(output.dtype, jnp.floating):
                raise TypeError(
                    "Structured nonlinear objective must return one real scalar."
                )
            return output

        objective, gradient = jax.value_and_grad(scalar)(point)
        constraints = _real_vector(
            self.constraints(point, args), self.num_constraints, "constraints"
        )
        jacobian = self.jacobian_plan.operator(point, args)
        finite = (
            jnp.isfinite(objective)
            & jnp.all(jnp.isfinite(gradient))
            & jnp.all(jnp.isfinite(constraints))
            & jnp.all(jnp.isfinite(jacobian.coefficients))
        )
        return StructuredNonlinearEvaluation(
            point,
            objective,
            gradient,
            constraints,
            jacobian,
            finite,
        )

    def hessian_operator(
        self,
        coordinates: ArrayLike,
        constraint_multipliers: ArrayLike,
        objective_factor: ArrayLike,
        args: Any = None,
        /,
    ) -> SparseCoordinateOperator:
        if self.hessian_plan is None:
            raise ValueError(
                "This structured nonlinear program has no exact Hessian plan."
            )
        point = self.validate_coordinates(coordinates)
        multipliers = _real_vector(
            constraint_multipliers,
            self.num_constraints,
            "constraint_multipliers",
        )
        factor = jnp.asarray(objective_factor, dtype=point.dtype)
        if factor.shape != ():
            raise ValueError("objective_factor must be scalar.")
        return self.hessian_plan.operator(point, (args, factor, multipliers))

    def warm_start(
        self,
        primal: ArrayLike,
        constraint_multipliers: ArrayLike,
        lower_bound_multipliers: ArrayLike,
        upper_bound_multipliers: ArrayLike,
        /,
        *,
        numeric_version: Any = 0,
        source_result_id: str | None = None,
        source_backend: str | None = None,
    ) -> StructuredNonlinearWarmStart:
        warm = StructuredNonlinearWarmStart(
            primal,
            constraint_multipliers,
            lower_bound_multipliers,
            upper_bound_multipliers,
            structure_id=self.structure_id,
            numeric_version=numeric_version,
            source_result_id=source_result_id,
            source_program_id=self.program_id,
            source_backend=source_backend,
        )
        if warm.constraint_multipliers.shape != (self.num_constraints,):
            raise ValueError(
                "warm-start constraint multipliers do not match the program."
            )
        return warm

    def _certificate(
        self,
        coordinates: ArrayLike,
        constraint_multipliers: ArrayLike,
        lower_bound_multipliers: ArrayLike,
        upper_bound_multipliers: ArrayLike,
        args: Any,
        variable_lower: Array,
        variable_upper: Array,
        constraint_lower: Array,
        constraint_upper: Array,
        /,
        *,
        active_tolerance: float,
    ) -> ConstrainedOptimalityCertificate:
        evaluation = self.evaluate(coordinates, args)
        multipliers = _real_vector(
            constraint_multipliers,
            self.num_constraints,
            "constraint_multipliers",
        )
        lower_dual = _real_vector(
            lower_bound_multipliers,
            self.num_variables,
            "lower_bound_multipliers",
        )
        upper_dual = _real_vector(
            upper_bound_multipliers,
            self.num_variables,
            "upper_bound_multipliers",
        )
        equality = multipliers[self.equality_indices]
        # A distinct two-sided row stores the net upper-minus-lower multiplier.
        # Recover its nonnegative parts; a one-sided row must retain a wrong sign
        # so the independent certificate can still reject an invalid multiplier.
        lower_net = -multipliers[self.lower_indices]
        upper_net = multipliers[self.upper_indices]
        lower_constraint_dual = jnp.where(
            jnp.isfinite(constraint_upper[self.lower_indices]),
            jnp.maximum(lower_net, 0.0),
            lower_net,
        )
        upper_constraint_dual = jnp.where(
            jnp.isfinite(constraint_lower[self.upper_indices]),
            jnp.maximum(upper_net, 0.0),
            upper_net,
        )
        inequality = jnp.concatenate((lower_constraint_dual, upper_constraint_dual))
        values = evaluation.constraints
        equality_values = values[self.equality_indices]
        equality_targets = constraint_lower[self.equality_indices]
        lower_slacks = values[self.lower_indices] - constraint_lower[self.lower_indices]
        upper_slacks = constraint_upper[self.upper_indices] - values[self.upper_indices]
        slacks = jnp.concatenate((lower_slacks, upper_slacks))
        stationarity = (
            evaluation.gradient
            + evaluation.jacobian.transpose_mv(multipliers)
            - lower_dual
            + upper_dual
        )
        lower_x_slack = evaluation.coordinates - variable_lower
        upper_x_slack = variable_upper - evaluation.coordinates
        lower_x_finite = jnp.isfinite(variable_lower)
        upper_x_finite = jnp.isfinite(variable_upper)
        primal = _maximum(jnp.abs(equality_values - equality_targets))
        primal = jnp.maximum(primal, _maximum(jnp.maximum(-lower_slacks, 0.0)))
        primal = jnp.maximum(primal, _maximum(jnp.maximum(-upper_slacks, 0.0)))
        primal = jnp.maximum(
            primal,
            _maximum(
                jnp.where(
                    lower_x_finite,
                    jnp.maximum(-lower_x_slack, 0.0),
                    0.0,
                )
            ),
        )
        primal = jnp.maximum(
            primal,
            _maximum(
                jnp.where(
                    upper_x_finite,
                    jnp.maximum(-upper_x_slack, 0.0),
                    0.0,
                )
            ),
        )
        dual = _maximum(jnp.maximum(-inequality, 0.0))
        dual = jnp.maximum(dual, _maximum(jnp.maximum(-lower_dual, 0.0)))
        dual = jnp.maximum(dual, _maximum(jnp.maximum(-upper_dual, 0.0)))
        complementarity = _maximum(jnp.abs(inequality * slacks))
        complementarity = jnp.maximum(
            complementarity,
            _maximum(
                jnp.where(
                    lower_x_finite,
                    jnp.abs(lower_dual * lower_x_slack),
                    0.0,
                )
            ),
        )
        complementarity = jnp.maximum(
            complementarity,
            _maximum(
                jnp.where(
                    upper_x_finite,
                    jnp.abs(upper_dual * upper_x_slack),
                    0.0,
                )
            ),
        )
        tolerance = float(active_tolerance)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("active_tolerance must be finite and non-negative.")
        active = slacks <= tolerance
        equality_sources = tuple(
            self.constraint_sources[int(index)]
            for index in np.asarray(self.equality_indices)
        )
        inequality_sources = tuple(
            self.constraint_sources[int(index)]
            for index in np.concatenate(
                (np.asarray(self.lower_indices), np.asarray(self.upper_indices))
            )
        )
        return ConstrainedOptimalityCertificate(
            equality_multipliers=equality,
            inequality_multipliers=inequality,
            slacks=slacks,
            active_mask=active,
            stationarity_residual=stationarity,
            primal_feasibility=primal,
            dual_feasibility=dual,
            complementarity=complementarity,
            equality_sources=equality_sources,
            inequality_sources=inequality_sources,
        )

    def certificate(
        self,
        coordinates: ArrayLike,
        constraint_multipliers: ArrayLike,
        lower_bound_multipliers: ArrayLike,
        upper_bound_multipliers: ArrayLike,
        args: Any = None,
        /,
        *,
        active_tolerance: float,
    ) -> ConstrainedOptimalityCertificate:
        return self._certificate(
            coordinates,
            constraint_multipliers,
            lower_bound_multipliers,
            upper_bound_multipliers,
            args,
            self.variable_lower,
            self.variable_upper,
            self.constraint_lower,
            self.constraint_upper,
            active_tolerance=active_tolerance,
        )


def _argument_signature(args: Any, /) -> str:
    leaves, structure = jax.tree_util.tree_flatten(args)
    records = []
    for leaf in leaves:
        if isinstance(leaf, (jax.Array, np.ndarray, jax.core.Tracer)):
            records.append(
                {
                    "kind": "array",
                    "shape": tuple(int(size) for size in leaf.shape),
                    "dtype": str(leaf.dtype),
                }
            )
        else:
            records.append(
                {
                    "kind": "static",
                    "type": f"{type(leaf).__module__}.{type(leaf).__qualname__}",
                    "value": repr(leaf),
                }
            )
    return canonical_fingerprint(
        {
            "kind": "structured-nonlinear-arguments",
            "structure": str(structure),
            "leaves": records,
        }
    )


def _bound_roles(
    variable_lower: Array,
    variable_upper: Array,
    constraint_lower: Array,
    constraint_upper: Array,
    /,
) -> tuple[np.ndarray, ...]:
    lower_x = np.asarray(variable_lower)
    upper_x = np.asarray(variable_upper)
    lower_c = np.asarray(constraint_lower)
    upper_c = np.asarray(constraint_upper)
    equality = np.isfinite(lower_c) & np.isfinite(upper_c) & (lower_c == upper_c)
    return (
        np.isfinite(lower_x),
        np.isfinite(upper_x),
        np.isfinite(lower_x) & np.isfinite(upper_x) & (lower_x == upper_x),
        equality,
        np.isfinite(lower_c) & ~equality,
        np.isfinite(upper_c) & ~equality,
    )


def _validated_bounds(
    value: ArrayLike | None,
    default: Array,
    size: int,
    owner: str,
    /,
) -> Array:
    result = default if value is None else _real_vector(value, size, owner)
    return jnp.asarray(result, dtype=default.dtype)


class StructuredNonlinearTemplate(StrictModule):
    """Coefficient-independent topology for one structured nonlinear family."""

    program: StructuredNonlinearProgram
    variable_lower_finite: Array
    variable_upper_finite: Array
    fixed_variable_mask: Array
    equality_mask: Array
    constraint_lower_finite: Array
    constraint_upper_finite: Array
    argument_signature: str = eqx.field(static=True)
    template_id: str = eqx.field(static=True)

    def __init__(
        self,
        program: StructuredNonlinearProgram,
        sample_args: Any = None,
        /,
    ):
        if not isinstance(program, StructuredNonlinearProgram):
            raise TypeError("program must be a StructuredNonlinearProgram.")
        roles = _bound_roles(
            program.variable_lower,
            program.variable_upper,
            program.constraint_lower,
            program.constraint_upper,
        )
        signature = _argument_signature(sample_args)
        self.program = program
        (
            variable_lower,
            variable_upper,
            fixed,
            equality,
            constraint_lower,
            constraint_upper,
        ) = (jnp.asarray(role, dtype=bool) for role in roles)
        self.variable_lower_finite = variable_lower
        self.variable_upper_finite = variable_upper
        self.fixed_variable_mask = fixed
        self.equality_mask = equality
        self.constraint_lower_finite = constraint_lower
        self.constraint_upper_finite = constraint_upper
        self.argument_signature = signature
        self.template_id = canonical_fingerprint(
            {
                "kind": "structured-nonlinear-template",
                "program": program.program_id,
                "structure": program.structure_id,
                "arguments": signature,
                "jacobian": program.jacobian_plan.plan_id,
                "hessian": (
                    None if program.hessian_plan is None else program.hessian_plan.plan_id
                ),
                "roles": tuple(role.tolist() for role in roles),
            }
        )


class PreparedStructuredNonlinearProgram(StrictModule):
    """Numeric structured nonlinear data bound to one reusable topology."""

    template: StructuredNonlinearTemplate
    args: Any
    variable_lower: Array
    variable_upper: Array
    constraint_lower: Array
    constraint_upper: Array
    objective_scale: Array
    constraint_scale: Array
    numeric_version: Array
    numeric_binding_id: str = eqx.field(static=True)

    def __init__(
        self,
        template: StructuredNonlinearTemplate,
        args: Any = None,
        /,
        *,
        variable_lower: ArrayLike | None = None,
        variable_upper: ArrayLike | None = None,
        constraint_lower: ArrayLike | None = None,
        constraint_upper: ArrayLike | None = None,
        objective_scale: ArrayLike = 1.0,
        constraint_scale: ArrayLike | None = None,
        numeric_version: Any = 0,
    ):
        if not isinstance(template, StructuredNonlinearTemplate):
            raise TypeError("template must be a StructuredNonlinearTemplate.")
        program = template.program
        if _argument_signature(args) != template.argument_signature:
            raise ValueError("Structured nonlinear argument structure changed.")
        lower_x = _validated_bounds(
            variable_lower,
            program.variable_lower,
            program.num_variables,
            "variable_lower",
        )
        upper_x = _validated_bounds(
            variable_upper,
            program.variable_upper,
            program.num_variables,
            "variable_upper",
        )
        lower_c = _validated_bounds(
            constraint_lower,
            program.constraint_lower,
            program.num_constraints,
            "constraint_lower",
        )
        upper_c = _validated_bounds(
            constraint_upper,
            program.constraint_upper,
            program.num_constraints,
            "constraint_upper",
        )
        if bool(np.any(np.asarray(lower_x) > np.asarray(upper_x))):
            raise ValueError("Variable lower bounds must not exceed upper bounds.")
        if bool(np.any(np.asarray(lower_c) > np.asarray(upper_c))):
            raise ValueError("Constraint lower bounds must not exceed upper bounds.")
        roles = _bound_roles(lower_x, upper_x, lower_c, upper_c)
        expected = (
            np.asarray(template.variable_lower_finite),
            np.asarray(template.variable_upper_finite),
            np.asarray(template.fixed_variable_mask),
            np.asarray(template.equality_mask),
            np.asarray(template.constraint_lower_finite),
            np.asarray(template.constraint_upper_finite),
        )
        if any(not np.array_equal(left, right) for left, right in zip(roles, expected)):
            raise ValueError("Structured nonlinear bound roles changed.")
        objective_scale_ = jnp.asarray(
            objective_scale,
            dtype=program.variable_lower.dtype,
        )
        if objective_scale_.shape != ():
            raise ValueError("objective_scale must be scalar.")
        objective_scale_ = eqx.error_if(
            objective_scale_,
            ~jnp.isfinite(objective_scale_) | (objective_scale_ <= 0.0),
            "objective_scale must be finite and positive.",
        )
        constraint_scale_ = (
            jnp.ones((program.num_constraints,), dtype=program.constraint_lower.dtype)
            if constraint_scale is None
            else _real_vector(
                constraint_scale,
                program.num_constraints,
                "constraint_scale",
            )
        )
        constraint_scale_ = eqx.error_if(
            constraint_scale_,
            jnp.any(~jnp.isfinite(constraint_scale_) | (constraint_scale_ <= 0.0)),
            "constraint_scale must be finite and positive.",
        )
        version = jnp.asarray(numeric_version, dtype=jnp.int32)
        if version.shape != ():
            raise ValueError("numeric_version must be scalar.")
        version = eqx.error_if(
            version,
            version < 0,
            "numeric_version must be non-negative.",
        )
        self.template = template
        self.args = args
        self.variable_lower = lower_x
        self.variable_upper = upper_x
        self.constraint_lower = lower_c
        self.constraint_upper = upper_c
        self.objective_scale = objective_scale_
        self.constraint_scale = jnp.asarray(
            constraint_scale_,
            dtype=program.constraint_lower.dtype,
        )
        self.numeric_version = version
        self.numeric_binding_id = canonical_fingerprint(
            {
                "kind": "prepared-structured-nonlinear",
                "template": template.template_id,
                "numeric_version": (
                    "traced"
                    if isinstance(version, jax.core.Tracer)
                    else int(np.asarray(version))
                ),
                "numerics": _numeric_fingerprint(
                    (
                        args,
                        lower_x,
                        upper_x,
                        lower_c,
                        upper_c,
                        objective_scale_,
                        constraint_scale_,
                    )
                ),
            }
        )

    @property
    def program(self) -> StructuredNonlinearProgram:
        return self.template.program

    @property
    def structure_id(self) -> str:
        return self.program.structure_id

    def validate_coordinates(self, coordinates: ArrayLike, /) -> Array:
        return self.program.validate_coordinates(coordinates)

    def evaluate(self, coordinates: ArrayLike, /) -> StructuredNonlinearEvaluation:
        return self.program.evaluate(coordinates, self.args)

    def hessian_operator(
        self,
        coordinates: ArrayLike,
        constraint_multipliers: ArrayLike,
        objective_factor: ArrayLike = 1.0,
        /,
    ) -> SparseCoordinateOperator:
        return self.program.hessian_operator(
            coordinates,
            constraint_multipliers,
            objective_factor,
            self.args,
        )

    def warm_start(
        self,
        primal: ArrayLike,
        constraint_multipliers: ArrayLike,
        lower_bound_multipliers: ArrayLike,
        upper_bound_multipliers: ArrayLike,
        /,
        *,
        source_result_id: str | None = None,
        source_backend: str | None = None,
    ) -> StructuredNonlinearWarmStart:
        return self.program.warm_start(
            primal,
            constraint_multipliers,
            lower_bound_multipliers,
            upper_bound_multipliers,
            numeric_version=self.numeric_version,
            source_result_id=source_result_id,
            source_backend=source_backend,
        )

    def certificate(
        self,
        coordinates: ArrayLike,
        constraint_multipliers: ArrayLike,
        lower_bound_multipliers: ArrayLike,
        upper_bound_multipliers: ArrayLike,
        /,
        *,
        active_tolerance: float,
    ) -> ConstrainedOptimalityCertificate:
        return self.program._certificate(
            coordinates,
            constraint_multipliers,
            lower_bound_multipliers,
            upper_bound_multipliers,
            self.args,
            self.variable_lower,
            self.variable_upper,
            self.constraint_lower,
            self.constraint_upper,
            active_tolerance=active_tolerance,
        )


class StructuredNonlinearResult(StrictModule):
    """Portable optimization result plus a backend-neutral structured warm start."""

    optimization: MinimizationResult
    warm_start: StructuredNonlinearWarmStart
    work: StructuredOptimizationWork
    numeric_version: Array
    structure_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        optimization: MinimizationResult,
        warm_start: StructuredNonlinearWarmStart,
        work: StructuredOptimizationWork,
        /,
        *,
        numeric_version: Any,
        structure_id: str,
        method_id: str,
    ):
        if not isinstance(optimization, MinimizationResult):
            raise TypeError("optimization must be a MinimizationResult.")
        if not isinstance(warm_start, StructuredNonlinearWarmStart):
            raise TypeError("warm_start must be a StructuredNonlinearWarmStart.")
        if not isinstance(work, StructuredOptimizationWork):
            raise TypeError("work must be StructuredOptimizationWork.")
        structure = _identifier(structure_id, "structure_id")
        method = _identifier(method_id, "method_id")
        if warm_start.structure_id != structure:
            raise ValueError("Warm-start structure does not match result structure.")
        version = jnp.asarray(numeric_version, dtype=jnp.int32)
        if version.shape != ():
            raise ValueError("numeric_version must be scalar.")
        version = eqx.error_if(
            version,
            version < 0,
            "numeric_version must be non-negative.",
        )
        self.optimization = optimization
        self.warm_start = warm_start
        self.work = work
        self.numeric_version = version
        self.structure_id = structure
        self.method_id = method
        self.result_id = canonical_fingerprint(
            {
                "kind": "structured-nonlinear-result",
                "structure": structure,
                "method": method,
                "warm_start": warm_start.warm_start_id,
            }
        )

    @property
    def successful(self) -> Array:
        return self.optimization.successful

    @property
    def parameters(self) -> Any:
        return self.optimization.parameters

    @property
    def objective(self) -> Array:
        return self.optimization.objective

    @property
    def status(self) -> Array:
        return self.optimization.status

    @property
    def certificate(self) -> Any:
        return self.optimization.certificate


def prepare_structured_template(
    program: StructuredNonlinearProgram,
    sample_args: Any = None,
    /,
) -> StructuredNonlinearTemplate:
    return StructuredNonlinearTemplate(program, sample_args)


def bind_structured_numeric(
    template: StructuredNonlinearTemplate,
    args: Any = None,
    /,
    **numeric: Any,
) -> PreparedStructuredNonlinearProgram:
    return PreparedStructuredNonlinearProgram(template, args, **numeric)


def prepare_structured_nonlinear(
    program: StructuredNonlinearProgram,
    args: Any = None,
    /,
    **numeric: Any,
) -> PreparedStructuredNonlinearProgram:
    template = prepare_structured_template(program, args)
    return bind_structured_numeric(template, args, **numeric)


def refresh_structured_nonlinear(
    prepared: PreparedStructuredNonlinearProgram,
    args: Any = _UNSET,
    /,
    **numeric: Any,
) -> PreparedStructuredNonlinearProgram:
    if not isinstance(prepared, PreparedStructuredNonlinearProgram):
        raise TypeError("prepared must be a PreparedStructuredNonlinearProgram.")
    refreshed_args = prepared.args if args is _UNSET else args
    return bind_structured_numeric(
        prepared.template,
        refreshed_args,
        variable_lower=numeric.pop("variable_lower", prepared.variable_lower),
        variable_upper=numeric.pop("variable_upper", prepared.variable_upper),
        constraint_lower=numeric.pop("constraint_lower", prepared.constraint_lower),
        constraint_upper=numeric.pop("constraint_upper", prepared.constraint_upper),
        objective_scale=numeric.pop("objective_scale", prepared.objective_scale),
        constraint_scale=numeric.pop("constraint_scale", prepared.constraint_scale),
        numeric_version=prepared.numeric_version + 1,
        **numeric,
    )


__all__ = [
    "PreparedStructuredNonlinearProgram",
    "StructuredNonlinearEvaluation",
    "StructuredNonlinearProgram",
    "StructuredNonlinearResult",
    "StructuredNonlinearTemplate",
    "StructuredNonlinearWarmStart",
    "StructuredOptimizationWork",
    "bind_structured_numeric",
    "prepare_structured_nonlinear",
    "prepare_structured_template",
    "refresh_structured_nonlinear",
]
