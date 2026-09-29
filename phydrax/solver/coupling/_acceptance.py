#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Execution and acceptance of prepared spatial coupled problems.

``solve_coupled_problem`` binds per-component runtime arguments, runs the
native linear (``phydrax.linalg.solve``) or nonlinear
(``phydrax.nonlinear.root``) solve, and then certifies the original
equations: every component's own residual (evaluated by its owner, plus the
law terms on its rows) and every law's interface defects (continuity and
flux balance from the owners' published fluxes). A result is accepted only
when the native solve succeeded and every certificate holds; finite values or
a small transformed residual never imply acceptance.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from ..._admissibility import guard_derivative_validity, refuse_derivative_dependencies
from ..._differentiation import DerivativeRoute, OwnerDerivativeCapability
from ..._strict import StrictModule
from ..._tree_math import uncancelled_direction
from ..._validation import positive_finite_float
from ...linalg import (
    AbstractInitialGuessProvider,
    DenseLU,
    InitialGuessDiagnostics,
    LinearDerivativeSolvePolicy,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSystem,
    plan,
    solve,
)
from ...measurement import PreparedQuantityField
from ...nonlinear import (
    NewtonKrylov,
    NonlinearResult,
    NonlinearTermination,
    root,
    select_initial_state,
)
from ._assembly import component_fields, component_residuals, owner_residual, OwnerValues
from ._laws import InterfaceDefectReport
from ._parameters import CoupledArguments
from ._prepared_problem import PreparedCoupledProblem


type CoupledInitialState = tuple[tuple[Array, ...], ...] | AbstractInitialGuessProvider


@final
class ComponentCertificate(StrictModule):
    """Original-equation residual of one component at the coupled solution.

    ``residual_norm`` is the norm of the owner's residual plus the law terms
    on its rows (including eliminated rows of another owner summed into the
    rows that express them), over the rows it keeps; its own eliminated rows
    are certified by the eliminating law's flux balance. ``scale`` sums the
    norms of the individual terms (owner operator action, data, law terms) and
    of the uncancelled term magnitude ``||J (s * u)||`` (fixed Rademacher signs
    ``s``), so an exact state whose terms cancel is measured against the terms
    it sums rather than against the cancelled sums themselves.
    """

    component: str = eqx.field(static=True)
    owner_id: str = eqx.field(static=True)
    residual_norm: Array
    scale: Array
    accepted: Array


@final
class CoupledSolution(StrictModule):
    """Evidence-carrying solution of one spatial coupled problem.

    ``state`` is in solve coordinates (``prepared.state_space``); ``fields``
    pairs every ``(component, field)`` with its full coefficients and
    ``law_states`` pairs every law that owns unknowns (for a mortar, the
    multiplier) with them. The keys are static, so a solution is a valid
    result of a compiled function. ``native_successful`` is the native solve
    status; ``accepted`` additionally requires every component certificate and
    every gated interface defect. ``derivative_valid`` is the native derivative
    admission combined with primal acceptance and a non-stopped parameter route;
    it stays separate from ``accepted``. It certifies that derivatives are
    admitted at an accepted primal solve, not the later derivative solve: a
    tangent or cotangent solve that fails is reported by the native linear
    contract, as NaN-poisoned derivatives under ``FailurePolicy("status")``.
    ``derivative_capability`` names the parameters the solve admits
    derivatives of and every refusal reason.
    ``initial_guess`` is the guarded-selection evidence of an initial-guess
    provider (proposal and native-baseline residual norms, and whether the
    proposal was used); the untrusted proposal never replaces ``state``.
    ``observation`` returns the predicted measurement of an observation binding
    at the certified fields; it is valid evidence only with ``accepted``.
    """

    state: tuple[tuple[Array, ...], ...]
    field_keys: tuple[tuple[str, str], ...] = eqx.field(static=True)
    field_values: tuple[Array, ...]
    law_ids: tuple[str, ...] = eqx.field(static=True)
    law_values: tuple[tuple[Array, ...], ...]
    observation_ids: tuple[str, ...] = eqx.field(static=True)
    observation_values: tuple[PreparedQuantityField, ...]
    linear: LinearSolveResult | None
    nonlinear: NonlinearResult | None
    initial_guess: InitialGuessDiagnostics | None
    components: tuple[ComponentCertificate, ...]
    interfaces: tuple[InterfaceDefectReport, ...]
    tolerance: float = eqx.field(static=True)
    native_successful: Array
    accepted: Array
    derivative_valid: Array
    derivative_capability: OwnerDerivativeCapability

    @property
    def fields(self) -> tuple[tuple[tuple[str, str], Array], ...]:
        return tuple(zip(self.field_keys, self.field_values, strict=True))

    @property
    def law_states(self) -> tuple[tuple[str, tuple[Array, ...]], ...]:
        return tuple(zip(self.law_ids, self.law_values, strict=True))

    def field(self, component: str, field: str, /) -> Array:
        for key, value in self.fields:
            if key == (component, field):
                return value
        raise KeyError(f"No field {field!r} of component {component!r}.")

    def law_state(self, law_id: str, /) -> tuple[Array, ...]:
        for key, value in self.law_states:
            if key == law_id:
                return value
        raise KeyError(f"Law {law_id!r} owns no unknowns.")

    def observation(self, binding_id: str, /) -> PreparedQuantityField:
        """Predicted measurement of one observation binding at this solution."""
        for key, value in zip(self.observation_ids, self.observation_values, strict=True):
            if key == binding_id:
                return value
        raise KeyError(f"No observation binding {binding_id!r}.")

    def interface(self, law_id: str, /) -> InterfaceDefectReport:
        for report in self.interfaces:
            if report.law_id == law_id:
                return report
        raise KeyError(f"No interface certificate of law {law_id!r}.")


def _norm(value: Array, /) -> Array:
    return jnp.sqrt(sum(jnp.sum(leaf**2) for leaf in jax.tree.leaves(value)))


def _kept(
    prepared: PreparedCoupledProblem, path: tuple[str, str], value: Array, /
) -> Array:
    positions = prepared.chart.kept_positions(path, rows=True)
    return value if positions is None else value[positions]


def _term_rows(
    prepared: PreparedCoupledProblem,
    values: OwnerValues,
    args: Mapping[str, object],
    /,
) -> OwnerValues:
    """Coupled rows of the linearized operator along ``uncancelled_direction(values)``.

    Their norm is the root-mean-square magnitude of the individual terms the
    coupled rows sum (owner operator including lifts, and law terms), which
    does not cancel when the solution does. Affine problems use the difference
    ``R(s * u) - R(0)`` (exact for affine rows and valid for owners that publish
    only reverse-mode rules); nonlinear problems the Newton JVP at ``u``. The
    scale is evidence, not a differentiable output.
    """
    chart = prepared.chart
    direction = uncancelled_direction(values)

    def rows(value: OwnerValues) -> OwnerValues:
        return chart.accumulate_rows(owner_residual(chart, value, args))

    match prepared.execution:
        case "linear":
            zero = rows({path: jnp.zeros_like(value) for path, value in values.items()})
            signed = rows(direction)
            terms = {path: signed[path] - zero[path] for path in signed}
        case "nonlinear":
            terms = jax.jvp(rows, (values,), (direction,))[1]
        case _:
            raise ValueError(f"Unknown coupled execution {prepared.execution!r}.")
    return jax.lax.stop_gradient(terms)


def _component_certificates(
    prepared: PreparedCoupledProblem,
    values: OwnerValues,
    args: Mapping[str, object],
    tolerance: float,
    /,
) -> tuple[ComponentCertificate, ...]:
    chart = prepared.chart
    native_rows = component_residuals(chart, values, args)
    rows = chart.accumulate_rows(owner_residual(chart, values, args, native=native_rows))
    zero_values = {path: jnp.zeros_like(value) for path, value in values.items()}
    zero_rows = component_residuals(chart, zero_values, args)
    terms = _term_rows(prepared, values, args)
    certificates: list[ComponentCertificate] = []
    for component in chart.components:
        native = tuple(
            native_rows[(component.name, b.name)] for b in component.row_blocks
        )
        at_zero = tuple(zero_rows[(component.name, b.name)] for b in component.row_blocks)
        residual = jnp.zeros((), dtype=jnp.float64)
        scale = jnp.zeros((), dtype=jnp.float64)
        for block, own, offset in zip(component.row_blocks, native, at_zero, strict=True):
            path = (component.name, block.name)
            coupled = _kept(prepared, path, rows[path])
            law_terms = coupled - _kept(prepared, path, own)
            residual = residual + jnp.sum(coupled**2)
            scale = (
                scale
                + _norm(_kept(prepared, path, own - offset))
                + _norm(_kept(prepared, path, offset))
                + _norm(law_terms)
                + _norm(_kept(prepared, path, terms[path]))
            )
        norm = jnp.sqrt(residual)
        certificates.append(
            ComponentCertificate(
                component=component.name,
                owner_id=component.owner_id,
                residual_norm=norm,
                scale=scale,
                accepted=jnp.isfinite(norm) & (norm <= tolerance * scale),
            )
        )
    return tuple(certificates)


def _default_linear_policy(
    system: LinearSystem, initial_state: CoupledInitialState | None, /
) -> LinearSolvePolicy | None:
    """The owner's linear policy when the caller declares none.

    Coupled saddle systems are indefinite, where an independent restarted
    Krylov derivative solve can stagnate and poison the derivative. When a
    dense direct solve is feasible under linalg's default budgets (no
    initial state, which a direct solve does not take, and no declared
    nullspace), the default is ``DenseLU`` whose implicit tangent and
    cotangent solves reuse its factors
    (``LinearDerivativeSolvePolicy(route="primal-factors")``). Otherwise
    ``None`` keeps linalg's capability-selected default.
    """
    if initial_state is not None or system.nullspace_policy is not None:
        return None
    direct = LinearSolvePolicy(
        DenseLU(),
        derivative_solve=LinearDerivativeSolvePolicy(route="primal-factors"),
    )
    try:
        plan(system, direct)
    except ValueError:
        # Dense materialization or factorization exceeds the default budgets.
        return None
    return direct


def _linear_solve(
    system: LinearSystem,
    rhs: tuple[tuple[Array, ...], ...],
    prepared: PreparedCoupledProblem,
    policy: LinearSolvePolicy | None,
    initial_state: CoupledInitialState | None,
    /,
) -> LinearSolveResult:
    if initial_state is None:
        return solve(system, rhs, policy=policy)
    guess = (
        initial_state
        if isinstance(initial_state, AbstractInitialGuessProvider)
        else prepared.state_space.validate(initial_state)
    )
    return solve(system, rhs, policy=policy, initial_guess=guess)


def _nonlinear_solve(
    prepared: PreparedCoupledProblem,
    args: Mapping[str, object],
    policy: LinearSolvePolicy | None,
    termination: NonlinearTermination | None,
    initial_state: CoupledInitialState | None,
    /,
) -> tuple[NonlinearResult, InitialGuessDiagnostics | None]:
    method = NewtonKrylov() if policy is None else NewtonKrylov(linear_policy=policy)
    problem = prepared.nonlinear_problem()
    evidence = None
    if initial_state is None:
        initial = prepared.state_space.zeros()
    elif isinstance(initial_state, AbstractInitialGuessProvider):
        initial, evidence = select_initial_state(
            problem, prepared.state_space.zeros(), initial_state, args=dict(args)
        )
    else:
        initial = initial_state
    result = root(
        problem,
        prepared.state_space.validate(initial),
        method=method,
        termination=termination,
        args=dict(args),
    )
    return result, evidence


def _owner_fields(
    prepared: PreparedCoupledProblem, values: OwnerValues, args: Mapping[str, object], /
) -> tuple[tuple[tuple[str, str], Array], ...]:
    fields = component_fields(prepared.chart, values, args)
    return tuple(
        ((component.name, record.name), fields[(component.name, record.name)])
        for component in prepared.chart.components
        for record in component.fields
    )


def _guard_parameter_derivatives(
    solution: CoupledSolution,
    prepared: PreparedCoupledProblem,
    bound: CoupledArguments,
    policy: LinearSolvePolicy | None,
    /,
) -> CoupledSolution:
    """Refuse non-admitted parameter derivatives; poison invalid admitted ones.

    The validity guard applies only when parameters are admitted: a problem
    without an admitted parameter route declares no derivative to invalidate.
    """
    capability = solution.derivative_capability
    parameters = bound.parameters
    admitted = tuple(
        value for name, value in parameters.items() if capability.admits(name)
    )
    refused = sorted(name for name in parameters if not capability.admits(name))
    reasons = dict(capability.refused)
    if refused:
        solution = refuse_derivative_dependencies(
            solution,
            tuple(parameters[name] for name in refused),
            message=(
                f"coupled problem {prepared.problem_id} refuses derivatives of "
                + "; ".join(f"{name!r} ({reasons[name]})" for name in refused)
            ),
        )
    if not admitted:
        return solution
    return guard_derivative_validity(
        solution,
        solution.derivative_valid,
        dependencies=admitted,
        failure="status" if policy is None else policy.failure.mode,
        message=(
            f"Derivatives of coupled problem {prepared.problem_id} require an "
            "accepted primal and a converged implicit derivative solve."
        ),
    )


def _refuse_raw_arguments(
    solution: CoupledSolution,
    prepared: PreparedCoupledProblem,
    bound: CoupledArguments,
    /,
) -> CoupledSolution:
    """Refuse derivatives through runtime arguments that are not bound parameters.

    A coupled solve publishes derivatives only through parameter bindings
    (``derivative_capability``): every inexact leaf of the runtime
    ``arguments`` that is not a bound parameter value (a raw coefficient, load,
    or learned model supplied by the caller) has no qualified route, so a
    derivative through it raises rather than returning a silent value.
    """
    bound_leaves = {id(leaf) for leaf in jax.tree.leaves(bound.parameter_values)}
    return refuse_derivative_dependencies(
        solution,
        tuple(
            leaf
            for leaf in jax.tree.leaves(bound.component_arguments)
            if id(leaf) not in bound_leaves
        ),
        message=(
            f"coupled problem {prepared.problem_id} admits derivatives only through "
            "parameter bindings; a runtime argument supplied outside a "
            "ParameterBinding has no qualified derivative route"
        ),
    )


def certify_coupled_state(
    prepared: PreparedCoupledProblem,
    state: tuple[tuple[Array, ...], ...],
    bound: CoupledArguments,
    /,
    *,
    policy: LinearSolvePolicy | None,
    linear: LinearSolveResult | None,
    nonlinear: NonlinearResult | None,
    native: Array,
    derivative: Array,
    tolerance: float,
    initial_guess: InitialGuessDiagnostics | None = None,
) -> CoupledSolution:
    """Certify one solve-coordinate state against the original equations.

    ``bound`` holds the per-component arguments the state was solved with and
    the bound parameter values; ``native`` is the native solve status and
    ``derivative`` the native derivative admission. Component residuals and
    interface defects decide ``accepted``; ``derivative_valid`` additionally
    requires the native admission and a non-stopped parameter route. Parameter
    derivatives the capability refuses raise ``derivative-unsupported`` when
    requested, and derivatives of a non-accepted solution are NaN (``"status"``
    failure mode) or raise (``"error"``). Observations are evaluated on the
    certified fields. ``initial_guess`` is the selection evidence of an
    initial-guess provider the solve started from.
    """
    if not isinstance(prepared, PreparedCoupledProblem):
        raise TypeError("prepared must be a PreparedCoupledProblem.")
    if not isinstance(bound, CoupledArguments):
        raise TypeError("bound must be CoupledArguments from prepared.bind_arguments.")
    tolerance_ = positive_finite_float(tolerance, "tolerance")
    args = bound.arguments
    chart = prepared.chart
    values = chart.owner_states(state, args)
    fields = _owner_fields(prepared, values, args)
    law_states = tuple(
        (
            law.law_id,
            tuple(values[(law.law_id, block.name)] for block in law.state_blocks),
        )
        for law in chart.laws
    )
    lookup = dict(fields)
    interfaces = tuple(
        law.certificate.defects(lookup, dict(law_states).get(law.law_id, ()), args)
        for law in chart.laws
    )
    components = _component_certificates(prepared, values, args, tolerance_)
    certified = jnp.all(jnp.stack([item.accepted for item in components])) & jnp.all(
        jnp.stack(
            [report.accepted(tolerance_) for report in interfaces] or [jnp.asarray(True)]
        )
    )
    accepted = jnp.asarray(native) & certified
    capability = prepared.derivative_capability(policy)
    routed = capability.derivative_contract.route is not DerivativeRoute.STOPPED
    solution = CoupledSolution(
        state=state,
        field_keys=tuple(key for key, _ in fields),
        field_values=tuple(value for _, value in fields),
        law_ids=tuple(key for key, _ in law_states),
        law_values=tuple(value for _, value in law_states),
        observation_ids=tuple(item.binding_id for item in prepared.observations),
        observation_values=tuple(
            item.evaluate(lookup, args) for item in prepared.observations
        ),
        linear=linear,
        nonlinear=nonlinear,
        initial_guess=initial_guess,
        components=components,
        interfaces=interfaces,
        tolerance=tolerance_,
        native_successful=jnp.asarray(native),
        accepted=accepted,
        derivative_valid=accepted & jnp.asarray(derivative) & routed,
        derivative_capability=capability,
    )
    return _guard_parameter_derivatives(solution, prepared, bound, policy)


def solve_coupled_problem(
    prepared: PreparedCoupledProblem,
    /,
    *,
    arguments: Mapping[str, object] | None = None,
    parameters: Mapping[str, object] | None = None,
    policy: LinearSolvePolicy | None = None,
    termination: NonlinearTermination | None = None,
    initial_state: CoupledInitialState | None = None,
    tolerance: float = 1.0e-8,
) -> CoupledSolution:
    """Solve a prepared coupled problem natively and certify its original equations.

    ``parameters`` binds a runtime value of every refresh parameter binding
    (arrays or ``ComponentBinding`` learned models) into the component
    ``arguments``; the prepared topology is never re-planned. Affine problems
    solve the assembled block ``LinearSystem`` with ``policy``; nonlinear
    problems run ``phydrax.nonlinear.root`` with a Newton--Krylov method whose
    inner solves use ``policy``. Without a ``policy``, an affine problem is
    solved by ``DenseLU`` with ``LinearDerivativeSolvePolicy(route=
    "primal-factors")``, so implicit derivatives of the indefinite coupled
    system reuse its factors, when a dense solve fits linalg's default budgets
    and neither an ``initial_state`` nor a nullspace is declared; otherwise
    linalg's capability-selected default applies. An explicit ``policy`` is
    used as given. ``initial_state`` is a solve-coordinate state
    used as given (its derivative follows the differentiation policy) or an
    ``AbstractInitialGuessProvider`` (an ``ACCELERATOR``): its untrusted
    proposal is used only when finite with a strictly smaller original residual
    than the native zero state, carries no derivative, and never enters
    acceptance, which certifies the solved state alone
    (``CoupledSolution.initial_guess`` reports the selection). Parameter
    derivatives are the implicit solution-map derivatives of the native solve's
    ``DifferentiationPolicy``, as published by
    ``prepared.derivative_capability(policy)``; a derivative through a runtime
    ``arguments`` leaf that no parameter binding supplies is refused
    (``derivative-unsupported``). ``tolerance`` is the relative acceptance
    tolerance of every certificate.
    """
    if not isinstance(prepared, PreparedCoupledProblem):
        raise TypeError("prepared must be a PreparedCoupledProblem.")
    bound = prepared.bind_arguments(arguments, parameters=parameters)
    args = bound.arguments
    match prepared.execution:
        case "linear":
            system, rhs = prepared.linear_system(args)
            policy_ = (
                _default_linear_policy(system, initial_state)
                if policy is None
                else policy
            )
            linear = _linear_solve(system, rhs, prepared, policy_, initial_state)
            solution = certify_coupled_state(
                prepared,
                prepared.state_space.validate(linear.value),
                bound,
                policy=policy_,
                linear=linear,
                nonlinear=None,
                native=linear.successful,
                derivative=linear.derivative_valid,
                tolerance=tolerance,
                initial_guess=linear.initial_guess,
            )
        case "nonlinear":
            nonlinear, evidence = _nonlinear_solve(
                prepared, args, policy, termination, initial_state
            )
            solution = certify_coupled_state(
                prepared,
                prepared.state_space.validate(nonlinear.state),
                bound,
                policy=policy,
                linear=None,
                nonlinear=nonlinear,
                native=nonlinear.successful,
                derivative=jnp.asarray(False),
                tolerance=tolerance,
                initial_guess=evidence,
            )
        case _:
            raise ValueError(f"Unknown coupled execution {prepared.execution!r}.")
    return _refuse_raw_arguments(solution, prepared, bound)


__all__ = [
    "ComponentCertificate",
    "CoupledSolution",
    "certify_coupled_state",
    "solve_coupled_problem",
]
