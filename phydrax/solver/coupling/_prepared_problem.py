#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Declaration and preparation of spatial coupled problems.

A :class:`CoupledProblemPlan` declares components, the interface bindings
their laws act on, the laws with their chosen impositions, gauge declarations,
parameter/observation bindings, and resource requests. Preparation validates
revision currency and identities, lowers every law through its components'
published capabilities, refuses duplicate boundary laws and undeclared
nullspaces, and retains canonical named state/row layouts for native linear
and nonlinear execution.
"""

from __future__ import annotations

import dataclasses
import enum
import types
from collections.abc import Mapping
from typing import final, Literal, TypeAlias

import equinox as eqx
import jax
import jax.extend.core as jex_core
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..._differentiation import DerivativeSurface, OwnerDerivativeCapability
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._validation import canonical_identifier, positive_finite_float
from ...linalg import (
    AbstractLinearOperator,
    BlockSpace,
    LinearSolvePolicy,
    LinearSubspace,
    LinearSystem,
    MaterializationPolicy,
    NullspacePolicy,
)
from ...linalg._subspaces import CompatibilityMode, GaugeMode
from ...nonlinear import NonlinearSystemProblem
from ...typing import parse
from ._assembly import (
    coupled_linear_system,
    coupled_nonlinear_problem,
    coupled_residual,
    coupled_weak_operator,
    CoupledChart,
    interface_lane_candidates,
    prepare_coupled_chart,
    require_square_owners,
    validate_contributions,
)
from ._components import AbstractSpatialComponent
from ._contributions import LawImposition, ResidualContribution
from ._execution import (
    CoupledExecutionPolicy,
    prepare_coupled_execution,
    PreparedCoupledExecution,
)
from ._interfaces import InterfaceBinding, InterfaceOwner
from ._kernel import detect_coupled_kernel
from ._laws import AbstractCouplingLaw, PreparedLaw
from ._observations import (
    AbstractObservationBinding,
    AbstractPreparedObservation,
    prepare_observations,
)
from ._parameters import (
    CoupledArguments,
    ParameterBinding,
    prepare_parameters,
    PreparedParameters,
)


CoupledExecution: TypeAlias = Literal["linear", "nonlinear"]


@final
class CoupledGauge(StrictModule):
    """Declared treatment of the kernel of a coupled operator.

    ``gauge`` and ``compatibility`` are the native ``NullspacePolicy`` modes
    applied to the kernel that preparation detects from the components'
    published kernels, completed through law-owned and interface-only
    unknowns. A coupled problem with a kernel and no gauge is refused.
    """

    gauge: GaugeMode = eqx.field(static=True)
    compatibility: CompatibilityMode = eqx.field(static=True)

    def __init__(
        self,
        gauge: GaugeMode = "minimum-norm",
        /,
        *,
        compatibility: CompatibilityMode = "error",
    ) -> None:
        self.gauge = parse(gauge, GaugeMode, "gauge")
        self.compatibility = parse(compatibility, CompatibilityMode, "compatibility")


@final
class CoupledResourcePolicy(StrictModule):
    """Resource requests of a coupled problem.

    ``materialization`` bounds any explicit materialization requested by the
    solve policy and the dense span images of kernel detection;
    ``kernel_tolerance`` is the relative threshold below which a direction of
    the detection span is a kernel of the coupled operator. ``execution``
    declares bounded signature-grouped lane execution of homogeneous
    components and contribution blocks; ``None`` executes every owner on its
    own and keeps the linear operator one named block grid.
    """

    materialization: MaterializationPolicy
    kernel_tolerance: float = eqx.field(static=True)
    execution: CoupledExecutionPolicy | None

    def __init__(
        self,
        *,
        materialization: MaterializationPolicy | None = None,
        kernel_tolerance: float = 1.0e-8,
        execution: CoupledExecutionPolicy | None = None,
    ) -> None:
        policy = MaterializationPolicy() if materialization is None else materialization
        if not isinstance(policy, MaterializationPolicy):
            raise TypeError("materialization must be a MaterializationPolicy.")
        if execution is not None and not isinstance(execution, CoupledExecutionPolicy):
            raise TypeError("execution must be a CoupledExecutionPolicy or None.")
        self.materialization = policy
        self.kernel_tolerance = positive_finite_float(
            kernel_tolerance, "kernel_tolerance"
        )
        self.execution = execution


def _identity(value: object, /) -> str:
    match value:
        case AbstractSpatialComponent():
            return value.name
        case AbstractCouplingLaw():
            return value.law_id
        case InterfaceBinding() | ParameterBinding() | AbstractObservationBinding():
            return value.binding_id
        case _:
            raise TypeError(f"{type(value).__name__} has no canonical plan identity.")


def _unique_sorted[T](values: tuple[T, ...], name: str, kind: type, /) -> tuple[T, ...]:
    """Validate kinds and unique identities; return canonical identity order."""
    if not isinstance(values, tuple) or not all(
        isinstance(value, kind) for value in values
    ):
        raise TypeError(f"{name} must be a tuple of {kind.__name__} values.")
    keyed = sorted(
        ((_identity(value), value) for value in values), key=lambda item: item[0]
    )
    if len({key for key, _ in keyed}) != len(keyed):
        raise ValueError(f"{name} identities must be unique.")
    return tuple(value for _, value in keyed)


@final
class CoupledProblemPlan(StrictModule):
    """Immutable declaration of one spatial coupled problem.

    Components are ordered by name, laws by law ID, and bindings by binding
    ID. Every law's interface binding must be declared, and every declared
    binding must be used by a law.
    """

    plan_id: str = eqx.field(static=True)
    components: tuple[AbstractSpatialComponent, ...]
    bindings: tuple[InterfaceBinding, ...]
    laws: tuple[AbstractCouplingLaw, ...]
    gauge: CoupledGauge | None
    parameters: tuple[ParameterBinding, ...]
    observations: tuple[AbstractObservationBinding, ...]
    resources: CoupledResourcePolicy

    def __init__(
        self,
        plan_id: str,
        /,
        *,
        components: tuple[AbstractSpatialComponent, ...],
        bindings: tuple[InterfaceBinding, ...],
        laws: tuple[AbstractCouplingLaw, ...],
        gauge: CoupledGauge | None = None,
        parameters: tuple[ParameterBinding, ...] = (),
        observations: tuple[AbstractObservationBinding, ...] = (),
        resources: CoupledResourcePolicy | None = None,
    ) -> None:
        identifier = canonical_identifier(plan_id, "plan_id")
        components_ = _unique_sorted(components, "components", AbstractSpatialComponent)
        bindings_ = _unique_sorted(bindings, "bindings", InterfaceBinding)
        laws_ = _unique_sorted(laws, "laws", AbstractCouplingLaw)
        parameters_ = _unique_sorted(parameters, "parameters", ParameterBinding)
        observations_ = _unique_sorted(
            observations, "observations", AbstractObservationBinding
        )
        if not components_:
            raise ValueError("A coupled problem has at least one component.")
        if gauge is not None and not isinstance(gauge, CoupledGauge):
            raise TypeError("gauge must be a CoupledGauge or None.")
        policy = CoupledResourcePolicy() if resources is None else resources
        if not isinstance(policy, CoupledResourcePolicy):
            raise TypeError("resources must be a CoupledResourcePolicy.")
        names = {component.name for component in components_}
        law_names = {law.law_id for law in laws_}
        if names & law_names:
            raise ValueError("Component names and law IDs share one namespace.")
        for binding in (*parameters_, *observations_):
            unknown = set(binding.components) - names
            if unknown:
                raise ValueError(
                    f"Binding {binding.binding_id!r} names unknown components {sorted(unknown)}."
                )
        declared = {binding.binding_id for binding in bindings_}
        used = {binding.binding_id for law in laws_ for binding in law.bindings}
        if used - declared:
            raise ValueError(
                "Every law's interface binding must be declared by the plan."
            )
        if declared - used:
            raise ValueError("Every declared interface binding must be used by a law.")
        self.plan_id = identifier
        self.components = components_
        self.bindings = bindings_
        self.laws = laws_
        self.gauge = gauge
        self.parameters = parameters_
        self.observations = observations_
        self.resources = policy


@final
class PreparedCoupledProblem(StrictModule):
    """Prepared spatial coupled problem with canonical named layouts.

    ``state_space``/``row_space`` are nested ``BlockSpace`` values addressed
    by ``(owner, block)`` paths: components (by name) then laws that own
    unknowns (by law ID). ``execution`` is ``"linear"`` when every component
    publishes an affine operator and every law contribution is affine.
    ``nullspace`` is the detected kernel of the coupled operator in solve
    coordinates, admitted only with a declared gauge. ``parameters`` holds the
    parameter bindings (with the values fixed at preparation) and
    ``observations`` the observations lowered through the owners' prepared
    queries, traces, and fluxes.
    """

    plan_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    chart: CoupledChart
    bindings: tuple[InterfaceBinding, ...]
    execution: CoupledExecution = eqx.field(static=True)
    nullspace_policy: NullspacePolicy | None
    parameters: PreparedParameters
    observations: tuple[AbstractPreparedObservation, ...]
    resources: CoupledResourcePolicy

    @property
    def components(self) -> tuple[AbstractSpatialComponent, ...]:
        return self.chart.components

    @property
    def laws(self) -> tuple[PreparedLaw, ...]:
        return self.chart.laws

    @property
    def state_space(self) -> BlockSpace:
        return self.chart.state_space

    @property
    def worksets(self) -> PreparedCoupledExecution | None:
        """Prepared lane worksets and their resource estimates (``None``: per owner)."""
        return self.chart.execution

    @property
    def row_space(self) -> BlockSpace:
        return self.chart.row_space

    def arguments(
        self, arguments: Mapping[str, object] | None = None, /
    ) -> dict[str, object]:
        """Per-component runtime arguments; omitted components receive ``None``."""
        supplied = {} if arguments is None else dict(arguments)
        names = [component.name for component in self.components]
        unknown = set(supplied) - set(names)
        if unknown:
            raise ValueError(f"Arguments name unknown components {sorted(unknown)}.")
        return {name: supplied.get(name) for name in names}

    def bind_arguments(
        self,
        arguments: Mapping[str, object] | None = None,
        /,
        *,
        parameters: Mapping[str, object] | None = None,
    ) -> CoupledArguments:
        """Bind every refresh parameter's runtime value into the component arguments.

        ``parameters`` maps binding IDs to arrays of the bound port's event shape
        or to ``ComponentBinding`` learned models; values fixed at preparation are
        refused here. The result's ``arguments`` feed ``residual``,
        ``linear_system``, and every other per-component-argument method.
        """
        names = tuple(component.name for component in self.components)
        return self.parameters.bind(names, arguments, parameters)

    def derivative_capability(
        self, policy: LinearSolvePolicy | None = None, /
    ) -> OwnerDerivativeCapability:
        """Bound parameters whose derivatives a solve under ``policy`` admits.

        Linear problems differentiate implicitly through the native solve:
        ``"mathematical"`` admits physical-parameter and solver-argument
        bindings, ``"rhs-only"`` only solver arguments; ``"algorithmic"`` and
        ``"none"``, nonlinear execution, re-preparation parameters, and prepared
        geometry, quadrature, and topology are refused with their reasons.
        """
        return self.parameters.derivative_capability(
            self.problem_id, self.execution == "linear", policy
        )

    def observation(self, binding_id: str, /) -> AbstractPreparedObservation:
        for observation in self.observations:
            if observation.binding_id == binding_id:
                return observation
        raise KeyError(f"No observation binding {binding_id!r}.")

    def residual(
        self,
        state: tuple[tuple[Array, ...], ...],
        arguments: Mapping[str, object] | None = None,
        /,
    ) -> tuple[tuple[Array, ...], ...]:
        """Coupled residual rows in solve coordinates."""
        return coupled_residual(
            self.chart, self.state_space.validate(state), self.arguments(arguments)
        )

    def linear_system(
        self, arguments: Mapping[str, object] | None = None, /
    ) -> tuple[LinearSystem, tuple[tuple[Array, ...], ...]]:
        """Native block ``LinearSystem`` and right-hand side of an affine problem.

        The operator is the named block endomorphism of ``state_space`` whose
        rows are the coupled residual rows identified with their paired state
        blocks by the inverse Riesz maps (Euclidean for FE/VEM coordinates).
        """
        if self.execution != "linear":
            raise ValueError("The coupled problem is nonlinear; use nonlinear_problem.")
        return coupled_linear_system(
            self.chart, self.arguments(arguments), nullspace_policy=self.nullspace_policy
        )

    def weak_operator(
        self, arguments: Mapping[str, object] | None = None, /
    ) -> AbstractLinearOperator:
        """Weak coupled Jacobian from ``state_space`` to ``row_space`` (affine problems)."""
        if self.execution != "linear":
            raise ValueError("The coupled problem is nonlinear; linearize its residual.")
        return coupled_weak_operator(self.chart, self.arguments(arguments))

    def nonlinear_problem(self) -> NonlinearSystemProblem:
        """Native ``NonlinearSystemProblem`` over the solve coordinates.

        Its ``args`` are the per-component argument mapping of ``arguments``;
        residual rows are identified with states as in ``linear_system``.
        """
        return coupled_nonlinear_problem(self.chart, self.problem_id)

    def owner_states(
        self,
        state: tuple[tuple[Array, ...], ...],
        arguments: Mapping[str, object] | None = None,
        /,
    ) -> dict[tuple[str, str], Array]:
        """Owner-coordinate blocks (component and law unknowns) of a solve state."""
        return self.chart.owner_states(state, self.arguments(arguments))

    def field(
        self,
        component: str,
        field: str,
        state: tuple[tuple[Array, ...], ...],
        arguments: Mapping[str, object] | None = None,
        /,
    ) -> Array:
        """Full coefficients of one component field at a solve state."""
        args = self.arguments(arguments)
        owner = self.chart.component(component)
        values = self.chart.owner_states(state, args)
        blocks = tuple(values[(component, block.name)] for block in owner.state_blocks)
        return owner.expand(field, blocks, args[component])


def _require_bindings(
    plan: CoupledProblemPlan, owners: tuple[InterfaceOwner, ...], /
) -> tuple[InterfaceOwner, ...]:
    """Current interface owners; every binding's endpoints are checked against them."""
    owners_ = tuple(owners)
    for binding in plan.bindings:
        binding.require_current(*owners_)
    return owners_


def _require_bound_law_inputs(
    laws: tuple[PreparedLaw, ...], bindings: tuple[ParameterBinding, ...], /
) -> None:
    """Every runtime input a law reads is the target of a refresh parameter binding.

    A law response (for example a learned potential) is admitted only through
    its binding's port and authority checks; the argument clash check then
    refuses a raw value supplied at the same key.
    """
    bound = {
        (target.component, target.name)
        for binding in bindings
        if binding.change == "refresh"
        for target in binding.targets
    }
    for law in laws:
        for contribution in law.contributions:
            if not isinstance(contribution, ResidualContribution):
                continue
            for item in contribution.residual.runtime_inputs:
                if (item.component, item.name) not in bound:
                    raise ValueError(
                        f"Law {law.law_id!r} reads runtime input {item.name!r} of "
                        f"component {item.component!r}, which no refresh "
                        "ParameterBinding targets; bind it so its value passes the "
                        "parameter's port and authority admission."
                    )


def _refuse_duplicate_laws(laws: tuple[PreparedLaw, ...], /) -> None:
    # Only impositions on one field of one component can overlap; bucketing by
    # that field keeps the check linear in the number of interfaces.
    buckets: dict[tuple[str, str, str], list[tuple[str, LawImposition]]] = {}
    for law in laws:
        for imposition in law.impositions:
            key = (
                imposition.component,
                imposition.field_space_id,
                imposition.entity_set_id,
            )
            buckets.setdefault(key, []).append((law.law_id, imposition))
    for claimed in buckets.values():
        for index, (law_id, imposition) in enumerate(claimed):
            for other_id, other in claimed[index + 1 :]:
                if imposition.overlaps(other):
                    raise ValueError(
                        f"Laws {law_id!r} and {other_id!r} both act on facets of field "
                        f"{imposition.field!r} of component {imposition.component!r}; "
                        "one boundary law is imposed once."
                    )


def _lane_execution(
    chart: CoupledChart,
    args: Mapping[str, object],
    execution: CoupledExecution,
    policy: CoupledExecutionPolicy,
    /,
) -> PreparedCoupledExecution:
    """Lane worksets of the components and contribution blocks without eliminated rows."""
    eligible = tuple(
        component
        for component in chart.components
        if all(
            chart.kept_positions((component.name, block.name)) is None
            for block in component.state_blocks
        )
        and all(
            chart.kept_positions((component.name, block.name), rows=True) is None
            for block in component.row_blocks
        )
    )
    interfaces = interface_lane_candidates(chart) if execution == "linear" else ()
    return prepare_coupled_execution(eligible, args, interfaces, policy)


def _execution(chart: CoupledChart, args: Mapping[str, object], /) -> CoupledExecution:
    if any(
        component.linear_operator(args[component.name]) is None
        for component in chart.components
    ):
        return "nonlinear"
    for law in chart.laws:
        for contribution in law.contributions:
            if isinstance(contribution, ResidualContribution) and not contribution.affine:
                return "nonlinear"
    return "linear"


def _nullspace_policy(
    chart: CoupledChart,
    gauge: CoupledGauge | None,
    args: Mapping[str, object],
    resources: CoupledResourcePolicy,
    /,
) -> NullspacePolicy | None:
    detected = detect_coupled_kernel(
        chart, args, resources.kernel_tolerance, resources.materialization
    )
    if detected is None:
        return None
    kernel, covectors = detected
    if gauge is None:
        raise ValueError(
            f"The coupled operator has a {kernel.shape[1]}-dimensional kernel built "
            "from the components' published kernels completed through law-owned and "
            "interface-only unknowns; declare a CoupledGauge."
        )
    # The linear system is K = M^{-1} A on the state space with inner product M.
    # Its range is M-orthogonal to ker(K*) = ker(M^{-1} A^T) = span(c) for the
    # row covectors c with c^T A = 0: f = M^{-1} g is compatible iff c^T g =
    # <c, f>_M = 0. The left kernel is therefore span(c) in state coordinates:
    # square owners pair row and state blocks positionally with equal sizes, so
    # the flattened row covectors are state coordinates (never M^{-1} c). The
    # basis is Euclidean-orthonormalized for conditioning only; projections use
    # the Gram matrix in the (generally non-Euclidean) state inner product.
    right = LinearSubspace(chart.state_space, jnp.asarray(kernel), orthonormal=True)
    left, _ = np.linalg.qr(covectors)
    left_space = LinearSubspace(chart.state_space, jnp.asarray(left))
    return NullspacePolicy(
        right=right,
        left=left_space,
        compatibility=gauge.compatibility,
        gauge=gauge.gauge,
    )


# Call-like primitives whose nested program receives the equation's inputs in order.
_CALL_PROGRAMS = ("jaxpr", "call_jaxpr", "fun_jaxpr")


def _nested_program(equation: jex_core.JaxprEqn, /) -> jex_core.Jaxpr | None:
    """The nested program of a call-like equation with one-to-one inputs."""
    for name in _CALL_PROGRAMS:
        program = equation.params.get(name)
        match program:
            case jex_core.ClosedJaxpr():
                inner = program.jaxpr
            case jex_core.Jaxpr():
                inner = program
            case _:
                continue
        if len(inner.invars) == len(equation.invars) and len(inner.outvars) == len(
            equation.outvars
        ):
            return inner
    return None


def _dependent_outputs(program: jex_core.Jaxpr, inputs: list[bool], /) -> list[bool]:
    """Outputs of ``program`` with a data dependence on the marked inputs.

    Call-like equations are followed into their programs; every other equation
    with a dependent input makes all its outputs dependent, so the result
    over-approximates the dependence and never misses one.
    """
    marked = {var for var, flag in zip(program.invars, inputs, strict=True) if flag}
    for equation in program.eqns:
        flags = [
            isinstance(var, jex_core.Var) and var in marked for var in equation.invars
        ]
        if not any(flags):
            continue
        inner = _nested_program(equation)
        outputs = (
            [True] * len(equation.outvars)
            if inner is None
            else _dependent_outputs(inner, flags)
        )
        marked.update(
            var for var, flag in zip(equation.outvars, outputs, strict=True) if flag
        )
    return [isinstance(var, jex_core.Var) and var in marked for var in program.outvars]


def _enters_operator(
    chart: CoupledChart,
    prepared: PreparedParameters,
    bind: tuple[tuple[str, ...], Mapping[str, object] | None, dict[str, object]],
    binding_id: str,
    probe: tuple[tuple[Array, ...], ...],
    /,
) -> bool:
    """Whether the coupled operator's action structurally depends on one parameter."""
    names, arguments, values = bind
    dynamic, static = eqx.partition(values[binding_id], eqx.is_inexact_array)
    leaves, treedef = jax.tree.flatten(dynamic)

    def action(*parameter: Array) -> Array:
        replaced = {
            **values,
            binding_id: eqx.combine(jax.tree.unflatten(treedef, parameter), static),
        }
        args = prepared.bind(names, arguments, replaced).arguments
        return chart.row_space.flatten(coupled_weak_operator(chart, args).mv(probe))

    program = jax.make_jaxpr(action)(*leaves).jaxpr
    return any(_dependent_outputs(program, [True] * len(program.invars)))


def _require_solver_arguments(
    chart: CoupledChart,
    prepared: PreparedParameters,
    bind: tuple[tuple[str, ...], Mapping[str, object] | None, dict[str, object]],
    /,
) -> None:
    """Refuse a solver-argument parameter that enters the coupled operator.

    Under ``"rhs-only"`` the native solve stops operator arrays, so a
    solver-argument declaration must be exact for every value: the traced
    program of the operator's action carries no data dependence on the
    parameter. A tangent at one reference value is not evidence (a dependence
    through ``p**2`` at ``p = 0``, or ``p[0] - p[1]`` along equal entries, has
    a vanishing tangent there).
    """
    probe = chart.state_space.unflatten(
        1.5 + jnp.cos(jnp.arange(chart.state_space.size, dtype=jnp.float64))
    )
    for binding in prepared.bindings:
        if binding.derivative is not DerivativeSurface.SOLVER_ARGUMENT:
            continue
        if _enters_operator(chart, prepared, bind, binding.binding_id, probe):
            raise ValueError(
                f"Parameter {binding.binding_id!r} declares a solver-argument surface "
                "but enters the coupled operator; declare it a physical parameter."
            )


def _declaration_payload(value: object, /) -> object:
    """Canonical content of a declaration: its fields, arrays, and static values.

    Modules contribute their type and every field; arrays their content
    fingerprint; scalars, strings, and enums their values. A callable that is
    not a module (an authored source or map function) is identified by its
    qualified type and name only, so replacing such a function without
    renaming the declaration's identifiers does not change the identity.
    """
    match value:
        case enum.Enum():
            return [type(value).__qualname__, value.name]
        case None | bool() | int() | str():
            return value
        case float() | complex():
            return repr(value)
        case jax.Array() | np.ndarray() | np.generic():
            return array_tree_fingerprint(np.asarray(value))
        case StrictModule():
            return {
                "type": f"{type(value).__module__}.{type(value).__qualname__}",
                "fields": {
                    field.name: _declaration_payload(getattr(value, field.name))
                    for field in dataclasses.fields(value)
                },
            }
        case tuple() | list():
            return [_declaration_payload(item) for item in value]
        case Mapping():
            return [
                [str(key), _declaration_payload(item)]
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            ]
        case types.FunctionType():
            return {"function": f"{value.__module__}.{value.__qualname__}"}
        case _:
            return {"opaque": f"{type(value).__module__}.{type(value).__qualname__}"}


def _prepared_law_payload(law: PreparedLaw, /) -> object:
    """Lowered content of one law: impositions, contribution routes, and arrays."""
    return {
        "law": law.law_id,
        "binding": law.binding_id,
        "impositions": _declaration_payload(law.impositions),
        "contributions": [
            [type(item).__qualname__, item.imposition_id] for item in law.contributions
        ],
        "arrays": array_tree_fingerprint(law),
    }


def prepare_coupled_problem(
    plan: CoupledProblemPlan,
    /,
    *,
    interface_owners: tuple[InterfaceOwner, ...] = (),
    arguments: Mapping[str, object] | None = None,
    parameters: Mapping[str, object] | None = None,
) -> PreparedCoupledProblem:
    """Validate and lower a coupled problem declaration once.

    ``interface_owners`` are the geometry/mesh owners the bindings are checked
    against at their exact revisions. ``arguments`` (per component) and
    ``parameters`` (a reference value of every parameter binding) evaluate the
    published kernels for the nullspace check and verify solver-argument
    declarations; only the values of ``"reprepare"`` parameters are retained,
    as fixed structure. Observation bindings are lowered once through their
    owners' prepared queries, traces, and fluxes.
    """
    if not isinstance(plan, CoupledProblemPlan):
        raise TypeError("plan must be a CoupledProblemPlan.")
    owners = _require_bindings(plan, interface_owners)
    components = {component.name: component for component in plan.components}
    laws = tuple(law.prepare(components, owners) for law in plan.laws)
    _refuse_duplicate_laws(laws)
    chart = prepare_coupled_chart(plan.components, laws)
    validate_contributions(chart)
    require_square_owners(chart)
    names = tuple(component.name for component in plan.components)
    prepared_parameters, values = prepare_parameters(plan.parameters, names, parameters)
    _require_bound_law_inputs(laws, plan.parameters)
    refresh: dict[str, object] = {
        key: value
        for key, value in values.items()
        if key not in prepared_parameters.fixed_ids
    }
    args = prepared_parameters.bind(names, arguments, refresh).arguments
    execution = _execution(chart, args)
    if plan.resources.execution is not None:
        chart = prepare_coupled_chart(
            plan.components,
            laws,
            execution=_lane_execution(chart, args, execution, plan.resources.execution),
        )
    policy = (
        _nullspace_policy(chart, plan.gauge, args, plan.resources)
        if execution == "linear"
        else None
    )
    if execution == "linear":
        _require_solver_arguments(chart, prepared_parameters, (names, arguments, refresh))
    problem_id = canonical_fingerprint(
        {
            "kind": "prepared-coupled-problem",
            "plan": plan.plan_id,
            "components": [component.owner_id for component in plan.components],
            "bindings": [binding.binding_id for binding in plan.bindings],
            "laws": [_declaration_payload(law) for law in plan.laws],
            "prepared_laws": [_prepared_law_payload(law) for law in laws],
            "gauge": _declaration_payload(plan.gauge),
            "resources": _declaration_payload(plan.resources),
            "parameters": [_declaration_payload(binding) for binding in plan.parameters],
            "fixed": array_tree_fingerprint(prepared_parameters.fixed_values),
            "observations": [binding.binding_id for binding in plan.observations],
            "state": chart.state_space.space_id,
            "rows": chart.row_space.space_id,
        }
    )
    return PreparedCoupledProblem(
        plan_id=plan.plan_id,
        problem_id=problem_id,
        chart=chart,
        bindings=plan.bindings,
        execution=execution,
        nullspace_policy=policy,
        parameters=prepared_parameters,
        observations=prepare_observations(plan.observations, components),
        resources=plan.resources,
    )


__all__ = [
    "CoupledExecution",
    "CoupledGauge",
    "CoupledProblemPlan",
    "CoupledResourcePolicy",
    "PreparedCoupledProblem",
    "prepare_coupled_problem",
]
