#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, PyTree

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..linalg import ArraySpace
from ..sparse import compile_sparse_hessian, compile_sparse_jacobian
from ._iterative import (
    ConstrainedOptimalityCertificate,
    MinimizationProblem,
    MinimizationResult,
    OptimizationTermination,
)
from ._structured_method import (
    AbstractStructuredNonlinearMethod,
    solve_structured_nonlinear,
)
from ._structured_nonlinear import (
    prepare_structured_nonlinear,
    PreparedStructuredNonlinearProgram,
    StructuredNonlinearProgram,
    StructuredNonlinearResult,
)


class StructuredMinimizationCompilation(StrictModule):
    """PyTree minimization problem lowered to one exact sparse bound-form NLP."""

    problem: MinimizationProblem
    program: StructuredNonlinearProgram
    prepared: PreparedStructuredNonlinearProgram
    initial_coordinates: Array
    unflatten: Any
    compilation_id: str = eqx.field(static=True)


class StructuredMinimizationResult(StrictModule):
    """Decoded PyTree minimizer paired with the underlying structured result."""

    optimization: MinimizationResult
    structured: StructuredNonlinearResult

    @property
    def successful(self) -> Array:
        return self.optimization.successful


def _constraint_values(
    problem: MinimizationProblem,
    parameters: PyTree[Any],
    args: Any,
    /,
) -> Array:
    values = []
    for constraint in problem.constraints:
        flat, _ = ravel_pytree(constraint.value(parameters, args))
        values.append(flat)
    flat_parameters, _ = ravel_pytree(parameters)
    return (
        jnp.concatenate(tuple(values))
        if values
        else jnp.empty((0,), dtype=flat_parameters.dtype)
    )


def _constraint_bounds(
    problem: MinimizationProblem,
    parameters: PyTree[Any],
    args: Any,
    /,
) -> tuple[Array, Array, tuple[str, ...]]:
    lower_values = []
    upper_values = []
    sources = []
    for constraint_index, constraint in enumerate(problem.constraints):
        value = constraint.value(parameters, args)
        lower, upper = constraint.bounds(value)
        lower_flat, _ = ravel_pytree(lower)
        upper_flat, _ = ravel_pytree(upper)
        lower_values.append(lower_flat)
        upper_values.append(upper_flat)
        sources.extend(
            f"constraint:{constraint_index}:{index}"
            for index in range(int(lower_flat.size))
        )
    flat_parameters, _ = ravel_pytree(parameters)
    if not lower_values:
        empty = jnp.empty((0,), dtype=flat_parameters.dtype)
        return empty, empty, ()
    return (
        jnp.concatenate(tuple(lower_values)),
        jnp.concatenate(tuple(upper_values)),
        tuple(sources),
    )


def compile_structured_minimization(
    problem: MinimizationProblem,
    initial_parameters: PyTree[Any],
    /,
    *,
    sample_args: Any = None,
    exact_hessian: bool = True,
    compiler: Any = "auto",
    chunk_size: int | None = None,
) -> StructuredMinimizationCompilation:
    """Lower one fixed-topology PyTree problem to reusable sparse derivatives."""
    if not isinstance(problem, MinimizationProblem):
        raise TypeError("problem must be a MinimizationProblem.")
    coordinates, unflatten = ravel_pytree(initial_parameters)
    source = ArraySpace((int(coordinates.size),), dtype=coordinates.dtype)
    lower_constraints, upper_constraints, sources = _constraint_bounds(
        problem,
        initial_parameters,
        sample_args,
    )
    target = ArraySpace(
        (int(lower_constraints.size),),
        dtype=coordinates.dtype,
    )
    if problem.bounds is None:
        variable_lower = jnp.full_like(coordinates, -jnp.inf)
        variable_upper = jnp.full_like(coordinates, jnp.inf)
    else:
        lower_parameters, upper_parameters = problem.bounds.materialize(
            initial_parameters
        )
        variable_lower = ravel_pytree(lower_parameters)[0]
        variable_upper = ravel_pytree(upper_parameters)[0]

    def objective(value, args):
        return problem.value(unflatten(value), args)[0]

    def constraints(value, args):
        return _constraint_values(problem, unflatten(value), args)

    jacobian = compile_sparse_jacobian(
        constraints,
        coordinates,
        source=source,
        target=target,
        sample_args=sample_args,
        compiler=compiler,
        chunk_size=chunk_size,
        plan_id=f"{problem.problem_id}:structured-jacobian",
    )
    hessian = None
    if exact_hessian:

        def lagrangian(value, packed):
            args, objective_factor, multipliers = packed
            return objective_factor * objective(value, args) + jnp.vdot(
                multipliers,
                constraints(value, args),
            )

        hessian = compile_sparse_hessian(
            lagrangian,
            coordinates,
            space=source,
            sample_args=(
                sample_args,
                jnp.asarray(1.0, dtype=coordinates.dtype),
                jnp.zeros((target.size,), dtype=coordinates.dtype),
            ),
            compiler=compiler,
            chunk_size=chunk_size,
            plan_id=f"{problem.problem_id}:structured-hessian",
        )
    structure_id = canonical_fingerprint(
        {
            "kind": "structured-minimization-compilation",
            "problem": problem.problem_id,
            "variables": int(coordinates.size),
            "constraints": int(lower_constraints.size),
            "jacobian": jacobian.plan_id,
            "hessian": None if hessian is None else hessian.plan_id,
        }
    )
    program = StructuredNonlinearProgram(
        objective,
        constraints,
        jacobian,
        variable_lower=variable_lower,
        variable_upper=variable_upper,
        constraint_lower=lower_constraints,
        constraint_upper=upper_constraints,
        constraint_sources=sources,
        hessian_plan=hessian,
        program_id=problem.problem_id,
        structure_id=structure_id,
    )
    prepared = prepare_structured_nonlinear(program, sample_args)
    return StructuredMinimizationCompilation(
        problem,
        program,
        prepared,
        coordinates,
        unflatten,
        structure_id,
    )


def _decoded_certificate(
    certificate: ConstrainedOptimalityCertificate | None,
    unflatten: Any,
    /,
) -> ConstrainedOptimalityCertificate | None:
    if certificate is None:
        return None
    return eqx.tree_at(
        lambda value: value.stationarity_residual,
        certificate,
        unflatten(jnp.asarray(certificate.stationarity_residual)),
    )


def solve_structured_minimization(
    compilation: StructuredMinimizationCompilation,
    /,
    *,
    method: AbstractStructuredNonlinearMethod,
    termination: OptimizationTermination | None = None,
    initial_parameters: PyTree[Any] | None = None,
    warm_start: Any = None,
) -> StructuredMinimizationResult:
    """Solve a compiled structured problem and decode its original PyTree."""
    if not isinstance(compilation, StructuredMinimizationCompilation):
        raise TypeError("compilation must be a StructuredMinimizationCompilation.")
    initial = (
        compilation.initial_coordinates
        if initial_parameters is None
        else ravel_pytree(initial_parameters)[0]
    )
    structured = solve_structured_nonlinear(
        compilation.prepared,
        initial,
        method=method,
        termination=termination,
        warm_start=warm_start,
    )
    raw = structured.optimization
    parameters = compilation.unflatten(raw.parameters)
    objective, auxiliary = compilation.problem.value(
        parameters,
        compilation.prepared.args,
    )
    optimization = MinimizationResult(
        parameters,
        objective,
        auxiliary,
        raw.status,
        raw.diagnostics,
        raw.provenance,
        certificate=_decoded_certificate(raw.certificate, compilation.unflatten),
        optimality_certificate=raw.optimality_certificate,
        status_evidence=raw.status_evidence,
        method_evidence=raw.method_evidence,
        precision_evidence=raw.precision_evidence,
    )
    return StructuredMinimizationResult(optimization, structured)


__all__ = [
    "StructuredMinimizationCompilation",
    "StructuredMinimizationResult",
    "compile_structured_minimization",
    "solve_structured_minimization",
]
