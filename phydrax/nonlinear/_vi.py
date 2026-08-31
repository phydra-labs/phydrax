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
from ..linalg import LinearSystem, PyTreeSpace, solve as solve_linear
from ._linearization import prepare_jacobian
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
    lower_metadata = problem.bounds._lower_metadata
    upper_metadata = problem.bounds._upper_metadata
    state_leaves = jax.tree.leaves(state)
    if lower_metadata is not None and upper_metadata is not None:

        def broadcast_metadata(metadata):
            if len(metadata) == 1 and metadata[0][0] == () and len(metadata[0][2]) == 1:
                scalar = metadata[0][2][0]
                return tuple(np.full(tuple(leaf.shape), scalar) for leaf in state_leaves)
            if len(metadata) != len(state_leaves):
                raise ValueError(
                    "Bound metadata does not match variational-inequality state."
                )
            values = []
            for item, leaf in zip(metadata, state_leaves, strict=True):
                shape, dtype, flattened = item
                value = np.asarray(flattened, dtype=np.dtype(dtype)).reshape(shape)
                values.append(np.broadcast_to(value, tuple(leaf.shape)))
            return tuple(values)

        lower_values = broadcast_metadata(lower_metadata)
        upper_values = broadcast_metadata(upper_metadata)
        roles = []
        for lower_host, upper_host in zip(lower_values, upper_values, strict=True):
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


class _ProjectedVISearch(StrictModule):
    state: PyTree[Array]
    residual: PyTree[Array]
    auxiliary: Any
    residual_norm: Array
    rate: Array
    evaluations: Array
    accepted: Array
    finite_seen: Array
    domain_failures: Array
    nonfinite_trials: Array


class _ProjectedVIRun(StrictModule):
    state: PyTree[Array]
    residual: PyTree[Array]
    auxiliary: Any
    initial_residual_norm: Array
    residual_norm: Array
    step_norm: Array
    iteration: Array
    residual_evaluations: Array
    jvp_evaluations: Array
    vjp_evaluations: Array
    jacobian_preparations: Array
    linear_solves: Array
    linear_iterations: Array
    accepted_steps: Array
    rejected_steps: Array
    domain_failures: Array
    nonfinite_trials: Array
    final_linear_status: Array
    final_linear_residual_norm: Array
    final_linear_converged: Array
    status: Array


def _solve_projected_semismooth(
    prepared: PreparedVariationalInequalitySolve,
    termination: NonlinearTermination,
    /,
) -> NonlinearResult:
    problem = prepared.problem
    method = prepared.method
    args = prepared.args
    nonlinear_problem = problem.as_nonlinear_problem(
        method.formulation,
        derivative_policy=method.derivative_policy,
        project_trials=False,
    )
    state = problem.bounds.project(prepared.nonlinear.state)
    residual, auxiliary = nonlinear_problem.evaluate(state, args)
    initial_norm = tree_norm(residual)
    finite = tree_allfinite(state) & tree_allfinite(residual)
    valid = nonlinear_problem.valid(state, residual, auxiliary, args)
    converged = (
        finite & valid & (initial_norm <= termination.residual_threshold(initial_norm))
    )
    run = _ProjectedVIRun(
        state=state,
        residual=residual,
        auxiliary=auxiliary,
        initial_residual_norm=initial_norm,
        residual_norm=initial_norm,
        step_norm=jnp.asarray(0.0, dtype=initial_norm.dtype),
        iteration=jnp.asarray(0, dtype=jnp.int32),
        residual_evaluations=jnp.asarray(1, dtype=jnp.int32),
        jvp_evaluations=jnp.asarray(0, dtype=jnp.int32),
        vjp_evaluations=jnp.asarray(0, dtype=jnp.int32),
        jacobian_preparations=jnp.asarray(0, dtype=jnp.int32),
        linear_solves=jnp.asarray(0, dtype=jnp.int32),
        linear_iterations=jnp.asarray(0, dtype=jnp.int32),
        accepted_steps=jnp.asarray(0, dtype=jnp.int32),
        rejected_steps=jnp.asarray(0, dtype=jnp.int32),
        domain_failures=(finite & ~valid).astype(jnp.int32),
        nonfinite_trials=(~finite).astype(jnp.int32),
        final_linear_status=jnp.asarray(-1, dtype=jnp.int32),
        final_linear_residual_norm=jnp.asarray(
            jnp.nan,
            dtype=initial_norm.dtype,
        ),
        final_linear_converged=jnp.asarray(False),
        status=jnp.where(
            converged,
            int(NonlinearStatus.SUCCESS),
            jnp.where(
                finite & valid,
                int(NonlinearStatus.ITERATING),
                jnp.where(
                    finite,
                    int(NonlinearStatus.UNRECOVERABLE_DOMAIN_FAILURE),
                    int(NonlinearStatus.NONFINITE_INPUT),
                ),
            ),
        ).astype(jnp.int32),
    )

    def condition(current):
        within_evaluations = (
            jnp.asarray(True)
            if termination.maximum_evaluations is None
            else current.residual_evaluations < termination.maximum_evaluations
        )
        within_linear = (
            jnp.asarray(True)
            if termination.maximum_linear_iterations is None
            else current.linear_iterations < termination.maximum_linear_iterations
        )
        return (
            (current.status == int(NonlinearStatus.ITERATING))
            & (current.iteration < termination.maximum_steps)
            & within_evaluations
            & within_linear
        )

    def body(current):
        jacobian = prepare_jacobian(
            nonlinear_problem,
            current.state,
            method.newton.jacobian_policy,
            args,
        )
        right_hand_side = jax.tree.map(jnp.negative, jacobian.residual)
        linear_result = solve_linear(
            LinearSystem(jacobian.operator),
            right_hand_side,
            policy=method.newton.linear_policy,
        )
        newton_direction = linear_result.value
        newton_image = jacobian.operator.mv(newton_direction)
        newton_slope = jnp.real(
            PyTreeSpace(jacobian.residual).inner(
                jacobian.residual,
                newton_image,
            )
        )
        linear_usable = (
            linear_result.diagnostics.converged
            & tree_allfinite(newton_direction)
            & jnp.isfinite(newton_slope)
            & (newton_slope < 0.0)
        )
        if jacobian.operator.capabilities.adjoint:
            merit_gradient = jacobian.operator.adjoint_mv(jacobian.residual)
        else:
            merit_gradient = nonlinear_problem.state_space.unflatten(
                nonlinear_problem.residual_space.flatten(jacobian.residual)
            )
        fallback_direction = jax.tree.map(jnp.negative, merit_gradient)
        direction = jax.tree.map(
            lambda newton_value, fallback_value: jnp.where(
                linear_usable,
                newton_value,
                fallback_value,
            ),
            newton_direction,
            fallback_direction,
        )
        direction_image = jacobian.operator.mv(direction)
        slope = jnp.real(
            PyTreeSpace(jacobian.residual).inner(
                jacobian.residual,
                direction_image,
            )
        )
        initial_rate = jnp.asarray(
            method.newton.line_search.initial_rate,
            dtype=current.residual_norm.dtype,
        )
        search = _ProjectedVISearch(
            state=current.state,
            residual=current.residual,
            auxiliary=current.auxiliary,
            residual_norm=current.residual_norm,
            rate=initial_rate,
            evaluations=jnp.asarray(0, dtype=jnp.int32),
            accepted=jnp.asarray(False),
            finite_seen=jnp.asarray(False),
            domain_failures=jnp.asarray(0, dtype=jnp.int32),
            nonfinite_trials=jnp.asarray(0, dtype=jnp.int32),
        )

        def search_condition(item):
            within_evaluations = (
                jnp.asarray(True)
                if termination.maximum_evaluations is None
                else (
                    current.residual_evaluations
                    + jacobian.residual_evaluations
                    + item.evaluations
                    < termination.maximum_evaluations
                )
            )
            return (
                ~item.accepted
                & (item.evaluations < method.newton.line_search.maximum_steps)
                & (item.rate >= method.newton.line_search.minimum_rate)
                & within_evaluations
            )

        def search_body(item):
            raw = jax.tree.map(
                lambda value, delta: value + item.rate * delta,
                current.state,
                direction,
            )
            candidate = problem.bounds.project(raw)
            candidate_residual, candidate_auxiliary = nonlinear_problem.evaluate(
                candidate, args
            )
            candidate_norm = tree_norm(candidate_residual)
            candidate_finite = tree_allfinite(candidate) & tree_allfinite(
                candidate_residual
            )
            candidate_valid = nonlinear_problem.valid(
                candidate,
                candidate_residual,
                candidate_auxiliary,
                args,
            )
            merit = 0.5 * candidate_norm * candidate_norm
            current_merit = 0.5 * current.residual_norm * current.residual_norm
            projected_step = jax.tree.map(
                lambda new, old: new - old,
                candidate,
                current.state,
            )
            projected_nonzero = tree_norm(projected_step) > 0.0
            accepted = (
                candidate_finite
                & candidate_valid
                & projected_nonzero
                & (
                    merit
                    <= current_merit
                    + method.newton.line_search.sufficient_decrease
                    * item.rate
                    * jnp.minimum(slope, -1e-30)
                )
            )
            return _ProjectedVISearch(
                state=jax.tree.map(
                    lambda proposed, old: jnp.where(
                        accepted,
                        proposed,
                        old,
                    ),
                    candidate,
                    item.state,
                ),
                residual=jax.tree.map(
                    lambda proposed, old: jnp.where(
                        accepted,
                        proposed,
                        old,
                    ),
                    candidate_residual,
                    item.residual,
                ),
                auxiliary=jax.tree.map(
                    lambda proposed, old: jnp.where(
                        accepted,
                        proposed,
                        old,
                    ),
                    candidate_auxiliary,
                    item.auxiliary,
                ),
                residual_norm=jnp.where(
                    accepted,
                    candidate_norm,
                    item.residual_norm,
                ),
                rate=jnp.where(
                    accepted,
                    item.rate,
                    method.newton.line_search.contraction * item.rate,
                ),
                evaluations=item.evaluations + 1,
                accepted=accepted,
                finite_seen=item.finite_seen | (candidate_finite & candidate_valid),
                domain_failures=item.domain_failures
                + (candidate_finite & ~candidate_valid).astype(jnp.int32),
                nonfinite_trials=item.nonfinite_trials
                + (~candidate_finite).astype(jnp.int32),
            )

        search = jax.lax.while_loop(
            search_condition,
            search_body,
            search,
        )
        step = jax.tree.map(
            lambda new, old: new - old,
            search.state,
            current.state,
        )
        step_norm = tree_norm(step)
        converged = search.accepted & (
            search.residual_norm
            <= termination.residual_threshold(current.initial_residual_norm)
        )
        stagnated = (
            search.accepted
            & ~converged
            & (step_norm <= termination.step_threshold(tree_norm(current.state)))
        )
        linear_iterations = jnp.sum(
            linear_result.diagnostics.iterations,
            dtype=jnp.int32,
        )
        next_evaluations = (
            current.residual_evaluations
            + jacobian.residual_evaluations
            + search.evaluations
        )
        next_linear_iterations = current.linear_iterations + linear_iterations
        evaluations_exhausted = (
            jnp.asarray(False)
            if termination.maximum_evaluations is None
            else next_evaluations >= termination.maximum_evaluations
        )
        linear_exhausted = (
            jnp.asarray(False)
            if termination.maximum_linear_iterations is None
            else next_linear_iterations >= termination.maximum_linear_iterations
        )
        status = jnp.where(
            converged,
            int(NonlinearStatus.SUCCESS),
            jnp.where(
                stagnated,
                int(NonlinearStatus.RESIDUAL_STAGNATION),
                jnp.where(
                    evaluations_exhausted,
                    int(NonlinearStatus.MAXIMUM_EVALUATIONS_REACHED),
                    jnp.where(
                        linear_exhausted,
                        int(NonlinearStatus.MAXIMUM_LINEAR_ITERATIONS_REACHED),
                        jnp.where(
                            search.accepted,
                            int(NonlinearStatus.ITERATING),
                            jnp.where(
                                search.finite_seen,
                                int(NonlinearStatus.LINE_SEARCH_FAILED),
                                jnp.where(
                                    search.domain_failures > 0,
                                    int(NonlinearStatus.RECOVERABLE_DOMAIN_FAILURE),
                                    int(NonlinearStatus.NONFINITE_EVALUATION),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        return _ProjectedVIRun(
            state=jax.tree.map(
                lambda proposed, old: jnp.where(
                    search.accepted,
                    proposed,
                    old,
                ),
                search.state,
                current.state,
            ),
            residual=jax.tree.map(
                lambda proposed, old: jnp.where(
                    search.accepted,
                    proposed,
                    old,
                ),
                search.residual,
                current.residual,
            ),
            auxiliary=jax.tree.map(
                lambda proposed, old: jnp.where(
                    search.accepted,
                    proposed,
                    old,
                ),
                search.auxiliary,
                current.auxiliary,
            ),
            initial_residual_norm=current.initial_residual_norm,
            residual_norm=jnp.where(
                search.accepted,
                search.residual_norm,
                current.residual_norm,
            ),
            step_norm=step_norm,
            iteration=current.iteration + 1,
            residual_evaluations=next_evaluations,
            jvp_evaluations=current.jvp_evaluations
            + jnp.sum(
                linear_result.diagnostics.matvec_count,
                dtype=jnp.int32,
            )
            + 2,
            vjp_evaluations=current.vjp_evaluations
            + jnp.sum(
                linear_result.diagnostics.adjoint_matvec_count,
                dtype=jnp.int32,
            )
            + 1,
            jacobian_preparations=current.jacobian_preparations + 1,
            linear_solves=current.linear_solves + 1,
            linear_iterations=next_linear_iterations,
            accepted_steps=current.accepted_steps + search.accepted.astype(jnp.int32),
            rejected_steps=current.rejected_steps + (~search.accepted).astype(jnp.int32),
            domain_failures=current.domain_failures + search.domain_failures,
            nonfinite_trials=current.nonfinite_trials + search.nonfinite_trials,
            final_linear_status=linear_result.status,
            final_linear_residual_norm=(linear_result.diagnostics.residual_norm),
            final_linear_converged=(linear_result.diagnostics.converged),
            status=status,
        )

    run = jax.lax.while_loop(condition, body, run)
    exhausted_status = jnp.where(
        run.status == int(NonlinearStatus.ITERATING),
        int(NonlinearStatus.MAXIMUM_STEPS_REACHED),
        run.status,
    ).astype(jnp.int32)
    diagnostics = NonlinearDiagnostics(
        initial_residual_norm=run.initial_residual_norm,
        final_residual_norm=run.residual_norm,
        final_step_norm=run.step_norm,
        iterations=run.iteration,
        residual_evaluations=run.residual_evaluations,
        jvp_evaluations=run.jvp_evaluations,
        vjp_evaluations=run.vjp_evaluations,
        jacobian_preparations=run.jacobian_preparations,
        linear_solves=run.linear_solves,
        linear_iterations=run.linear_iterations,
        accepted_steps=run.accepted_steps,
        rejected_steps=run.rejected_steps,
        domain_failures=run.domain_failures,
        nonfinite_trials=run.nonfinite_trials,
        final_linear_status=run.final_linear_status,
        final_linear_residual_norm=run.final_linear_residual_norm,
        final_linear_converged=run.final_linear_converged,
    )
    return NonlinearResult(
        state=run.state,
        residual=run.residual,
        auxiliary=run.auxiliary,
        status=exhausted_status,
        diagnostics=diagnostics,
        provenance=NonlinearProvenance(
            problem_id=nonlinear_problem.problem_id,
            method_id="semismooth-newton/projected",
            derivative_id=method.newton.jacobian_policy.mode,
            globalization_id="projected-complementarity-armijo",
            notes="Operator evaluations are restricted to the closed box.",
        ),
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
    result = (
        _solve_projected_semismooth(prepared, termination_)
        if prepared.method.feasibility == "preserve-box"
        else solve_prepared_nonlinear(
            prepared.nonlinear,
            termination=termination_,
        )
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
        attempts=result.attempts,
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
