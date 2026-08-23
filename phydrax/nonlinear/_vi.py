#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from math import isfinite, sqrt
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PyTree

from .._bounds import Bounds
from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._tree_math import tree_allfinite, tree_norm, validate_real_inexact_tree
from ..linalg import PyTreeSpace
from ._newton import NewtonKrylov
from ._prepared import (
    prepare_nonlinear,
    PreparedNonlinearSolve,
    refresh_nonlinear,
    solve_prepared_nonlinear,
)
from ._types import (
    NonlinearDiagnostics,
    NonlinearProvenance,
    NonlinearResult,
    NonlinearStatus,
    NonlinearSystemProblem,
    NonlinearTermination,
)


ComplementarityFormulation: TypeAlias = Literal["natural", "fischer-burmeister"]
VariationalInequalityFeasibility: TypeAlias = Literal["allow-infeasible", "preserve-box"]


@jax.custom_jvp
def _fischer_burmeister(a: Array, b: Array, origin_coefficient: Array, /) -> Array:
    return jnp.hypot(a, b) - a - b


@_fischer_burmeister.defjvp
def _fischer_burmeister_jvp(primals, tangents):
    a, b, origin_coefficient = primals
    da, db, _ = tangents
    radius = jnp.hypot(a, b)
    coefficient_a = jnp.where(
        radius > 0.0,
        a / radius,
        origin_coefficient,
    )
    coefficient_b = jnp.where(
        radius > 0.0,
        b / radius,
        origin_coefficient,
    )
    primal = radius - a - b
    tangent = (coefficient_a - 1.0) * da + (coefficient_b - 1.0) * db
    return primal, tangent


class GeneralizedDerivativePolicy(StrictModule):
    """Explicit Clarke selection for Fischer--Burmeister origin points."""

    origin_coefficient: float = eqx.field(static=True)

    def __init__(self, *, origin_coefficient: float = 1.0 / sqrt(2.0)):
        value = float(origin_coefficient)
        if not isfinite(value) or 2.0 * value * value > 1.0:
            raise ValueError(
                "origin_coefficient must define a vector in the Clarke unit ball."
            )
        self.origin_coefficient = value


def _natural_map(
    value: PyTree[Any],
    operator_value: PyTree[Any],
    bounds: Bounds,
    /,
) -> PyTree[Array]:
    trial = jax.tree.map(lambda x, f: x - f, value, operator_value)
    projected = bounds.project(trial)
    return jax.tree.map(lambda x, y: x - y, value, projected)


def _fischer_burmeister_map(
    value: PyTree[Any],
    operator_value: PyTree[Any],
    bounds: Bounds,
    origin_coefficient: float,
    /,
) -> PyTree[Array]:
    lower, upper = bounds.materialize(value)

    def residual_leaf(x, f, lo, hi):
        coefficient = jnp.asarray(origin_coefficient, dtype=x.dtype)
        fixed = lo == hi
        lower_finite = jnp.isfinite(lo)
        upper_finite = jnp.isfinite(hi)
        lower_distance = jnp.where(lower_finite, x - lo, 0.0)
        upper_distance = jnp.where(upper_finite, hi - x, 0.0)
        lower_only = _fischer_burmeister(lower_distance, f, coefficient)
        upper_only = _fischer_burmeister(upper_distance, -f, coefficient)
        nested_box = _fischer_burmeister(
            lower_distance,
            _fischer_burmeister(upper_distance, -f, coefficient),
            coefficient,
        )
        return jnp.where(
            fixed,
            x - lo,
            jnp.where(
                lower_finite & upper_finite,
                nested_box,
                jnp.where(
                    lower_finite,
                    lower_only,
                    jnp.where(upper_finite, upper_only, f),
                ),
            ),
        )

    return jax.tree.map(residual_leaf, value, operator_value, lower, upper)


class VariationalInequalityProblem(StrictModule):
    """Box variational inequality ``F(x)`` with explicit feasible bounds."""

    operator: Callable[[PyTree[Any], Any], PyTree[Array]]
    bounds: Bounds
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        operator: Callable[[PyTree[Any], Any], PyTree[Array]],
        bounds: Bounds,
        /,
        *,
        problem_id: str = "variational-inequality",
    ):
        if not callable(operator):
            raise TypeError("operator must be callable.")
        if not isinstance(bounds, Bounds):
            raise TypeError("bounds must be Bounds.")
        identifier = str(problem_id)
        if not identifier:
            raise ValueError("problem_id must be non-empty.")
        self.operator = operator
        self.bounds = bounds
        self.problem_id = identifier

    def evaluate(
        self,
        state: PyTree[Any],
        args: Any = None,
        /,
    ) -> PyTree[Array]:
        value = validate_real_inexact_tree(state, name="VI state")
        operator_value = validate_real_inexact_tree(
            self.operator(value, args), name="VI operator value"
        )
        return PyTreeSpace(value).validate(operator_value)

    def natural_residual(
        self,
        state: PyTree[Any],
        args: Any = None,
        /,
    ) -> PyTree[Array]:
        value = validate_real_inexact_tree(state, name="VI state")
        return _natural_map(value, self.evaluate(value, args), self.bounds)

    def fischer_burmeister_residual(
        self,
        state: PyTree[Any],
        args: Any = None,
        /,
        *,
        policy: GeneralizedDerivativePolicy | None = None,
    ) -> PyTree[Array]:
        value = validate_real_inexact_tree(state, name="MCP state")
        policy_ = GeneralizedDerivativePolicy() if policy is None else policy
        if not isinstance(policy_, GeneralizedDerivativePolicy):
            raise TypeError("policy must be GeneralizedDerivativePolicy or None.")
        return _fischer_burmeister_map(
            value,
            self.evaluate(value, args),
            self.bounds,
            policy_.origin_coefficient,
        )

    def as_nonlinear_problem(
        self,
        formulation: ComplementarityFormulation = "fischer-burmeister",
        /,
        *,
        derivative_policy: GeneralizedDerivativePolicy | None = None,
        project_trials: bool = False,
    ) -> NonlinearSystemProblem:
        if formulation not in ("natural", "fischer-burmeister"):
            raise ValueError(f"Unknown complementarity formulation {formulation!r}.")
        policy = (
            GeneralizedDerivativePolicy()
            if derivative_policy is None
            else derivative_policy
        )
        if not isinstance(policy, GeneralizedDerivativePolicy):
            raise TypeError(
                "derivative_policy must be GeneralizedDerivativePolicy or None."
            )

        def residual(state, args):
            raw_value = validate_real_inexact_tree(state, name="VI state")
            value = self.bounds.project(raw_value) if project_trials else raw_value
            physical = self.evaluate(value, args)
            transformed = (
                _natural_map(value, physical, self.bounds)
                if formulation == "natural"
                else _fischer_burmeister_map(
                    value,
                    physical,
                    self.bounds,
                    policy.origin_coefficient,
                )
            )
            return transformed, physical

        return NonlinearSystemProblem(
            residual,
            has_aux=True,
            problem_id=(
                f"{self.problem_id}/{formulation}/projected"
                if project_trials
                else f"{self.problem_id}/{formulation}"
            ),
        )


class ComplementarityCertificate(StrictModule):
    """Feasibility, natural-map, and active-set evidence at a candidate solution."""

    feasibility_violation: Array
    natural_residual_norm: Array
    fischer_burmeister_norm: Array
    lower_active: Array
    upper_active: Array
    fixed: Array
    free: Array
    finite: Array
    feasible: Array
    complementary: Array
    certified: Array


class VariationalInequalityResult(StrictModule):
    """Semismooth nonlinear result paired with a complementarity certificate."""

    nonlinear_result: NonlinearResult
    certificate: ComplementarityCertificate

    @property
    def state(self) -> PyTree[Array]:
        return self.nonlinear_result.state

    @property
    def status(self) -> Array:
        return self.nonlinear_result.status

    @property
    def residual(self) -> PyTree[Array]:
        return self.nonlinear_result.residual

    @property
    def auxiliary(self) -> Any:
        return self.nonlinear_result.auxiliary

    @property
    def diagnostics(self) -> NonlinearDiagnostics:
        return self.nonlinear_result.diagnostics

    @property
    def provenance(self) -> NonlinearProvenance:
        return self.nonlinear_result.provenance

    @property
    def successful(self) -> Array:
        return self.nonlinear_result.successful


class SemismoothNewton(StrictModule):
    """Semismooth Newton solve for natural-map or Fischer--Burmeister equations."""

    newton: NewtonKrylov
    formulation: ComplementarityFormulation = eqx.field(static=True)
    derivative_policy: GeneralizedDerivativePolicy
    certification_tolerance: float = eqx.field(static=True)
    feasibility: VariationalInequalityFeasibility = eqx.field(static=True)

    def __init__(
        self,
        *,
        newton: NewtonKrylov | None = None,
        formulation: ComplementarityFormulation = "fischer-burmeister",
        feasibility: VariationalInequalityFeasibility = "allow-infeasible",
        derivative_policy: GeneralizedDerivativePolicy | None = None,
        certification_tolerance: float = 1e-7,
    ):
        newton_ = NewtonKrylov() if newton is None else newton
        policy_ = (
            GeneralizedDerivativePolicy()
            if derivative_policy is None
            else derivative_policy
        )
        tolerance = float(certification_tolerance)
        if not isinstance(newton_, NewtonKrylov):
            raise TypeError("newton must be NewtonKrylov or None.")
        if formulation not in ("natural", "fischer-burmeister"):
            raise ValueError(f"Unknown complementarity formulation {formulation!r}.")
        if feasibility not in ("allow-infeasible", "preserve-box"):
            raise ValueError("feasibility must be 'allow-infeasible' or 'preserve-box'.")
        if not isinstance(policy_, GeneralizedDerivativePolicy):
            raise TypeError(
                "derivative_policy must be GeneralizedDerivativePolicy or None."
            )
        if not isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("certification_tolerance must be finite and positive.")
        self.newton = newton_
        self.formulation = formulation
        self.derivative_policy = policy_
        self.certification_tolerance = tolerance
        self.feasibility = feasibility

    def solve(
        self,
        problem: VariationalInequalityProblem,
        initial_state: PyTree[Any],
        /,
        *,
        termination: NonlinearTermination | None = None,
        args: Any = None,
    ) -> VariationalInequalityResult:
        prepared = prepare_variational_inequality(
            problem,
            initial_state,
            method=self,
            termination=termination,
            args=args,
        )
        return solve_prepared_variational_inequality(prepared)


class PreparedVariationalInequalitySolve(StrictModule):
    """Prepared semismooth VI solve with fixed bound-role topology."""

    problem: VariationalInequalityProblem
    method: SemismoothNewton
    termination: NonlinearTermination
    args: Any
    nonlinear: PreparedNonlinearSolve
    numeric_version: Array
    topology_id: str = eqx.field(static=True)

    def __init__(
        self,
        problem: VariationalInequalityProblem,
        method: SemismoothNewton,
        termination: NonlinearTermination,
        args: Any,
        nonlinear: PreparedNonlinearSolve,
        /,
        *,
        topology_id: str,
        numeric_version: Any,
    ):
        if not isinstance(problem, VariationalInequalityProblem):
            raise TypeError("problem must be VariationalInequalityProblem.")
        if not isinstance(method, SemismoothNewton):
            raise TypeError("method must be SemismoothNewton.")
        if not isinstance(termination, NonlinearTermination):
            raise TypeError("termination must be NonlinearTermination.")
        if not isinstance(nonlinear, PreparedNonlinearSolve):
            raise TypeError("nonlinear must be PreparedNonlinearSolve.")
        topology_id_ = str(topology_id)
        if not topology_id_:
            raise ValueError("topology_id must be non-empty.")
        version = jnp.asarray(numeric_version, dtype=jnp.int32)
        if version.ndim != 0:
            raise ValueError("numeric_version must be scalar.")
        version = eqx.error_if(
            version,
            version < 0,
            "numeric_version must be non-negative.",
        )
        self.problem = problem
        self.method = method
        self.termination = termination
        self.args = args
        self.nonlinear = nonlinear
        self.numeric_version = version
        self.topology_id = topology_id_


def _bound_topology_id(problem: VariationalInequalityProblem, state, /) -> str:
    lower, upper = problem.bounds.materialize(state)
    roles = []
    for lower_leaf, upper_leaf in zip(
        jax.tree.leaves(lower),
        jax.tree.leaves(upper),
        strict=True,
    ):
        lower_host = np.asarray(lower_leaf)
        upper_host = np.asarray(upper_leaf)
        role = (
            np.isfinite(lower_host).astype(np.int8)
            + 2 * np.isfinite(upper_host).astype(np.int8)
            + 4
            * (
                np.isfinite(lower_host)
                & np.isfinite(upper_host)
                & (lower_host == upper_host)
            ).astype(np.int8)
        )
        roles.append(
            {
                "shape": list(role.shape),
                "roles": role.reshape(-1).tolist(),
            }
        )
    return canonical_fingerprint(
        {
            "kind": "variational-inequality-bound-topology",
            "problem": problem.problem_id,
            "roles": roles,
        }
    )


def _vi_nonlinear_problem(
    problem: VariationalInequalityProblem,
    method: SemismoothNewton,
    /,
) -> NonlinearSystemProblem:
    return problem.as_nonlinear_problem(
        method.formulation,
        derivative_policy=method.derivative_policy,
        project_trials=method.feasibility == "preserve-box",
    )


def prepare_variational_inequality(
    problem: VariationalInequalityProblem,
    initial_state: PyTree[Any],
    /,
    *,
    method: SemismoothNewton | None = None,
    termination: NonlinearTermination | None = None,
    args: Any = None,
) -> PreparedVariationalInequalitySolve:
    if not isinstance(problem, VariationalInequalityProblem):
        raise TypeError("problem must be VariationalInequalityProblem.")
    method_ = SemismoothNewton() if method is None else method
    termination_ = NonlinearTermination() if termination is None else termination
    if not isinstance(method_, SemismoothNewton):
        raise TypeError("method must be SemismoothNewton or None.")
    if not isinstance(termination_, NonlinearTermination):
        raise TypeError("termination must be NonlinearTermination or None.")
    initial = problem.bounds.project(initial_state)
    nonlinear = prepare_nonlinear(
        _vi_nonlinear_problem(problem, method_),
        initial,
        method=method_.newton,
        termination=termination_,
        args=args,
    )
    return PreparedVariationalInequalitySolve(
        problem,
        method_,
        termination_,
        args,
        nonlinear,
        topology_id=_bound_topology_id(problem, initial),
        numeric_version=0,
    )


def refresh_variational_inequality(
    prepared: PreparedVariationalInequalitySolve,
    problem: VariationalInequalityProblem,
    initial_state: PyTree[Any],
    /,
    *,
    args: Any = None,
) -> PreparedVariationalInequalitySolve:
    if not isinstance(prepared, PreparedVariationalInequalitySolve):
        raise TypeError("prepared must be PreparedVariationalInequalitySolve.")
    if not isinstance(problem, VariationalInequalityProblem):
        raise TypeError("problem must be VariationalInequalityProblem.")
    if problem.problem_id != prepared.problem.problem_id:
        raise ValueError("VI refresh must preserve problem_id.")
    initial = problem.bounds.project(initial_state)
    topology_id = _bound_topology_id(problem, initial)
    if topology_id != prepared.topology_id:
        raise ValueError("VI refresh changed finite/infinite/fixed bound topology.")
    nonlinear = refresh_nonlinear(
        prepared.nonlinear,
        _vi_nonlinear_problem(problem, prepared.method),
        initial,
        args=args,
    )
    return PreparedVariationalInequalitySolve(
        problem,
        prepared.method,
        prepared.termination,
        args,
        nonlinear,
        topology_id=prepared.topology_id,
        numeric_version=prepared.numeric_version + 1,
    )


def solve_prepared_variational_inequality(
    prepared: PreparedVariationalInequalitySolve,
    /,
    *,
    termination: NonlinearTermination | None = None,
) -> VariationalInequalityResult:
    if not isinstance(prepared, PreparedVariationalInequalitySolve):
        raise TypeError("prepared must be PreparedVariationalInequalitySolve.")
    termination_ = prepared.termination if termination is None else termination
    if not isinstance(termination_, NonlinearTermination):
        raise TypeError("termination must be NonlinearTermination or None.")
    result = solve_prepared_nonlinear(
        prepared.nonlinear,
        termination=termination_,
    )
    return _certify_variational_inequality_result(
        prepared.problem,
        prepared.method,
        result,
        prepared.args,
    )


def _certify_variational_inequality_result(
    problem: VariationalInequalityProblem,
    method: SemismoothNewton,
    result: NonlinearResult,
    args: Any,
    /,
) -> VariationalInequalityResult:
    state = (
        problem.bounds.project(result.state)
        if method.feasibility == "preserve-box"
        else result.state
    )
    nonlinear_problem = _vi_nonlinear_problem(problem, method)
    transformed, physical = nonlinear_problem.evaluate(state, args)
    certificate = complementarity_certificate(
        problem,
        state,
        args=args,
        tolerance=method.certification_tolerance,
        derivative_policy=method.derivative_policy,
    )
    certified_status = jnp.where(
        (result.status == int(NonlinearStatus.SUCCESS)) & ~certificate.certified,
        int(NonlinearStatus.RESIDUAL_STAGNATION),
        result.status,
    ).astype(jnp.int32)
    diagnostics = eqx.tree_at(
        lambda value: (
            value.final_residual_norm,
            value.residual_evaluations,
        ),
        result.diagnostics,
        (
            tree_norm(transformed),
            result.diagnostics.residual_evaluations + 1,
        ),
    )
    provenance = NonlinearProvenance(
        problem_id=nonlinear_problem.problem_id,
        method_id=result.provenance.method_id,
        derivative_id=result.provenance.derivative_id,
        globalization_id=result.provenance.globalization_id,
        linear_plan_id=result.provenance.linear_plan_id,
        notes=(f"vi-formulation={method.formulation};feasibility={method.feasibility}"),
    )
    certified_result = NonlinearResult(
        state=state,
        residual=transformed,
        auxiliary=physical,
        status=certified_status,
        diagnostics=diagnostics,
        provenance=provenance,
    )
    return VariationalInequalityResult(
        nonlinear_result=certified_result,
        certificate=certificate,
    )


def complementarity_certificate(
    problem: VariationalInequalityProblem,
    state: PyTree[Any],
    /,
    *,
    args: Any = None,
    tolerance: float = 1e-7,
    derivative_policy: GeneralizedDerivativePolicy | None = None,
) -> ComplementarityCertificate:
    """Evaluate a pointwise box-complementarity certificate."""
    if not isinstance(problem, VariationalInequalityProblem):
        raise TypeError("problem must be a VariationalInequalityProblem.")
    tolerance_ = float(tolerance)
    if not isfinite(tolerance_) or tolerance_ <= 0.0:
        raise ValueError("tolerance must be finite and positive.")
    policy = (
        GeneralizedDerivativePolicy() if derivative_policy is None else derivative_policy
    )
    if not isinstance(policy, GeneralizedDerivativePolicy):
        raise TypeError("derivative_policy must be GeneralizedDerivativePolicy or None.")
    value = validate_real_inexact_tree(state, name="certificate state")
    operator_value = problem.evaluate(value, args)
    lower, upper = problem.bounds.materialize(value)
    natural = _natural_map(value, operator_value, problem.bounds)
    fischer = _fischer_burmeister_map(
        value,
        operator_value,
        problem.bounds,
        policy.origin_coefficient,
    )
    lower_active = jnp.asarray(0, dtype=jnp.int32)
    upper_active = jnp.asarray(0, dtype=jnp.int32)
    fixed = jnp.asarray(0, dtype=jnp.int32)
    for x, lo, hi in zip(
        jax.tree.leaves(value),
        jax.tree.leaves(lower),
        jax.tree.leaves(upper),
        strict=True,
    ):
        fixed_mask = lo == hi
        lower_mask = ~fixed_mask & jnp.isfinite(lo) & (x <= lo + tolerance_)
        upper_mask = ~fixed_mask & ~lower_mask & jnp.isfinite(hi) & (x >= hi - tolerance_)
        lower_active = lower_active + jnp.sum(lower_mask, dtype=jnp.int32)
        upper_active = upper_active + jnp.sum(upper_mask, dtype=jnp.int32)
        fixed = fixed + jnp.sum(fixed_mask, dtype=jnp.int32)
    total = sum(leaf.size for leaf in jax.tree.leaves(value))
    free = jnp.asarray(total, dtype=jnp.int32) - lower_active - upper_active - fixed
    feasibility = problem.bounds.violation(value)
    natural_norm = tree_norm(natural)
    fischer_norm = tree_norm(fischer)
    finite = (
        jnp.isfinite(feasibility)
        & jnp.isfinite(natural_norm)
        & jnp.isfinite(fischer_norm)
        & tree_allfinite(operator_value)
    )
    feasible = jnp.isfinite(feasibility) & (feasibility <= tolerance_)
    complementary = finite & (natural_norm <= tolerance_) & (fischer_norm <= tolerance_)
    certified = feasible & complementary
    return ComplementarityCertificate(
        feasibility_violation=feasibility,
        natural_residual_norm=natural_norm,
        fischer_burmeister_norm=fischer_norm,
        lower_active=lower_active,
        upper_active=upper_active,
        fixed=fixed,
        free=free,
        finite=finite,
        feasible=feasible,
        complementary=complementary,
        certified=certified,
    )


__all__ = [
    "ComplementarityCertificate",
    "ComplementarityFormulation",
    "PreparedVariationalInequalitySolve",
    "GeneralizedDerivativePolicy",
    "SemismoothNewton",
    "VariationalInequalityProblem",
    "VariationalInequalityResult",
    "VariationalInequalityFeasibility",
    "complementarity_certificate",
    "prepare_variational_inequality",
    "refresh_variational_inequality",
    "solve_prepared_variational_inequality",
]
