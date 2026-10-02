#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Parameter bindings of spatial coupled problems.

A :class:`ParameterBinding` binds one scientific input, identified by a
``ValuePort`` (semantic identity, event shape, representation, and declared
physical dimensions), to named native runtime inputs of one or more
components. It states the parameter's role, the derivative surface through
which it enters the coupled map, and whether a change is a numeric refresh of
the prepared problem or requires preparing it again.

Prepared topology, layouts, interface quadrature, and routes stay fixed; bound
parameter values are dynamic runtime arguments supplied at every solve. A
value is an array of the port's event shape or a separately held learned model
bound through a ``ComponentBinding`` whose owner ports publish the parameter
port. Values whose change requires re-preparation are fixed at preparation and
refused at solve time.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import final, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from ..._differentiation import (
    ComponentAuthority,
    DerivativeRoute,
    DerivativeSurface,
    OwnerDerivativeCapability,
)
from ..._model._component import ComponentBinding
from ..._model._ports import ValuePort
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import canonical_identifier
from ...linalg import LinearSolvePolicy
from ...linalg._policies import DifferentiationMode
from ...typing import checked, parse


ParameterRole: TypeAlias = Literal["coefficient", "source", "boundary", "control"]
ParameterChange: TypeAlias = Literal["refresh", "reprepare"]

# Structure fixed by preparation; its derivatives are refused by every coupled solve.
_FIXED_STRUCTURE = {
    "geometry": (
        "mesh coordinates, interface matching, and common refinements are fixed "
        "prepared geometry"
    ),
    "quadrature": "interface and owner quadrature rules are fixed prepared structure",
    "topology": "component topology, DOF layouts, and constraint rows are fixed",
}
_CONDITIONS = ("accepted-result", "prepared-geometry-fixed", "solve-converged")
_LEARNED_AUTHORITIES = (ComponentAuthority.MODEL, ComponentAuthority.DISCRETIZATION)
_ROLE_SURFACES: dict[ParameterRole, DerivativeSurface | None] = {
    "coefficient": DerivativeSurface.PHYSICAL_PARAMETER,
    "source": DerivativeSurface.SOLVER_ARGUMENT,
    "boundary": None,
    "control": None,
}


@final
class RuntimeInput(StrictModule, NonTrainableState):
    """One named native runtime input of one component.

    ``name`` is the key under which the component's owner reads the value from
    its runtime user arguments (finite-element coefficients read
    ``context.user_args[name]``, virtual-element coefficients ``args[name]``).
    """

    component: str = eqx.field(static=True)
    name: str = eqx.field(static=True)

    def __init__(self, component: str, name: str, /) -> None:
        self.component = canonical_identifier(component, "component")
        self.name = canonical_identifier(name, "name")


@final
class ParameterBinding(StrictModule, NonTrainableState):
    """Explicit binding of one scientific parameter to native runtime inputs.

    ``port`` is the parameter's scientific identity; its event shape is the
    array shape of a bound value. ``role`` states the physical meaning:
    material ``"coefficient"`` values enter the coupled operator
    (``PHYSICAL_PARAMETER``), ``"source"`` values its right-hand side
    (``SOLVER_ARGUMENT``); ``"boundary"`` and ``"control"`` inputs declare
    whichever surface they enter. ``derivative`` is the surface a consumer
    differentiates through (``None``: no derivative is requested). A
    ``"reprepare"`` change fixes the value at preparation and admits no
    derivative.
    """

    binding_id: str = eqx.field(static=True)
    components: tuple[str, ...] = eqx.field(static=True)
    targets: tuple[RuntimeInput, ...]
    port: ValuePort
    role: ParameterRole = eqx.field(static=True)
    derivative: DerivativeSurface | None = eqx.field(static=True)
    change: ParameterChange = eqx.field(static=True)

    @checked
    def __init__(
        self,
        binding_id: str,
        port: ValuePort,
        /,
        *,
        targets: tuple[RuntimeInput, ...],
        role: ParameterRole,
        derivative: DerivativeSurface | None = None,
        change: ParameterChange = "refresh",
    ) -> None:
        identifier = canonical_identifier(binding_id, "binding_id")
        if (
            not isinstance(targets, tuple)
            or not targets
            or not all(isinstance(target, RuntimeInput) for target in targets)
        ):
            raise TypeError("targets must be a nonempty tuple of RuntimeInput values.")
        keys = [(target.component, target.name) for target in targets]
        if len(set(keys)) != len(keys):
            raise ValueError(f"Parameter {identifier!r} names one runtime input twice.")
        role_ = parse(role, ParameterRole, "role")
        change_ = parse(change, ParameterChange, "change")
        _require_surface(identifier, role_, derivative, change_)
        self.binding_id = identifier
        self.components = tuple(sorted({target.component for target in targets}))
        self.targets = tuple(
            sorted(targets, key=lambda item: (item.component, item.name))
        )
        self.port = port
        self.role = role_
        self.derivative = derivative
        self.change = change_


def _require_surface(
    binding_id: str,
    role: ParameterRole,
    derivative: DerivativeSurface | None,
    change: ParameterChange,
    /,
) -> None:
    if derivative is None:
        return
    if derivative not in (
        DerivativeSurface.PHYSICAL_PARAMETER,
        DerivativeSurface.SOLVER_ARGUMENT,
    ):
        raise ValueError(
            f"Parameter {binding_id!r} enters a coupled solve as a physical parameter "
            f"or a solver argument, not as {derivative.value!r}."
        )
    required = _ROLE_SURFACES[role]
    if required is not None and derivative is not required:
        raise ValueError(
            f"A {role} parameter enters the coupled map as {required.value!r}; "
            f"{binding_id!r} declares {derivative.value!r}."
        )
    if change == "reprepare":
        raise ValueError(
            f"Parameter {binding_id!r} requires re-preparation when it changes and "
            "admits no derivative surface."
        )


type ParameterValue = Array | ComponentBinding


def _checked_value(binding: ParameterBinding, value: object, /) -> ParameterValue:
    """Validate one bound value against its port and admitted authorities."""
    if isinstance(value, ComponentBinding):
        if binding.change == "reprepare":
            raise ValueError(
                f"Parameter {binding.binding_id!r} is fixed at preparation; a "
                "learned model is a runtime value."
            )
        if value.authority not in _LEARNED_AUTHORITIES:
            raise ValueError(
                f"A component with {value.authority.value} authority cannot supply "
                f"parameter {binding.binding_id!r}: a learned physical input changes the "
                "accepted equations and requires model or discretization authority."
            )
        outputs = () if value.owner_ports is None else value.owner_ports.outputs
        if binding.port.port_id not in {port.port_id for port in outputs}:
            raise ValueError(
                f"The learned model bound to parameter {binding.binding_id!r} does "
                "not publish its port among its owner output ports."
            )
        return value
    array = jnp.asarray(value)
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        raise TypeError(f"Parameter {binding.binding_id!r} values must be inexact.")
    if array.shape != binding.port.event_shape:
        raise ValueError(
            f"Parameter {binding.binding_id!r} has event shape "
            f"{binding.port.event_shape}; received {array.shape}."
        )
    return array


def _owner_value(value: ParameterValue, /) -> object:
    """The runtime input an owner reads: an array or the bound model itself."""
    return value.model if isinstance(value, ComponentBinding) else value


@final
class CoupledArguments(StrictModule):
    """Per-component runtime arguments with the parameter values bound into them.

    ``arguments`` is the mapping consumed by the prepared problem's residual,
    operator, and solve methods. ``parameters`` pairs every bound parameter
    with its value (fixed ones included) so evidence and derivative guards can
    name the dependencies of a result.
    """

    component_names: tuple[str, ...] = eqx.field(static=True)
    component_arguments: tuple[object, ...]
    parameter_ids: tuple[str, ...] = eqx.field(static=True)
    parameter_values: tuple[ParameterValue, ...]

    @property
    def arguments(self) -> dict[str, object]:
        return dict(zip(self.component_names, self.component_arguments, strict=True))

    @property
    def parameters(self) -> dict[str, ParameterValue]:
        return dict(zip(self.parameter_ids, self.parameter_values, strict=True))


def _component_inputs(
    base: object, component: str, inputs: Mapping[str, object], /
) -> object:
    if not inputs:
        return base
    if base is None:
        return dict(inputs)
    if not isinstance(base, Mapping):
        raise TypeError(
            f"Component {component!r} receives parameters through its runtime user "
            f"arguments; supply them as a mapping, not {type(base).__name__}."
        )
    clash = set(base) & set(inputs)
    if clash:
        raise ValueError(
            f"Runtime inputs {sorted(clash)} of component {component!r} are bound "
            "parameters; supply them through parameters, not arguments."
        )
    return {**base, **inputs}


@final
class PreparedParameters(StrictModule):
    """Parameter bindings of a prepared coupled problem.

    ``fixed_values`` hold the values of ``"reprepare"`` bindings, which are
    fixed structure of the prepared problem; every ``"refresh"`` binding is
    supplied explicitly at each solve and never retained.
    """

    bindings: tuple[ParameterBinding, ...]
    fixed_ids: tuple[str, ...] = eqx.field(static=True)
    fixed_values: tuple[Array, ...]

    def binding(self, binding_id: str, /) -> ParameterBinding:
        for binding in self.bindings:
            if binding.binding_id == binding_id:
                return binding
        raise KeyError(f"No parameter binding {binding_id!r}.")

    def bind(
        self,
        names: tuple[str, ...],
        arguments: Mapping[str, object] | None,
        parameters: Mapping[str, object] | None,
        /,
    ) -> CoupledArguments:
        """Bind runtime values of every refresh parameter into component arguments."""
        supplied = {} if arguments is None else dict(arguments)
        unknown = set(supplied) - set(names)
        if unknown:
            raise ValueError(f"Arguments name unknown components {sorted(unknown)}.")
        values = self._values({} if parameters is None else dict(parameters))
        inputs: dict[str, dict[str, object]] = {name: {} for name in names}
        for binding in self.bindings:
            for target in binding.targets:
                inputs[target.component][target.name] = _owner_value(
                    values[binding.binding_id]
                )
        component_arguments = tuple(
            _component_inputs(supplied.get(name), name, inputs[name]) for name in names
        )
        ordered = tuple(binding.binding_id for binding in self.bindings)
        return CoupledArguments(
            component_names=names,
            component_arguments=component_arguments,
            parameter_ids=ordered,
            parameter_values=tuple(values[name] for name in ordered),
        )

    def _values(self, supplied: Mapping[str, object], /) -> dict[str, ParameterValue]:
        declared = {binding.binding_id for binding in self.bindings}
        unknown = set(supplied) - declared
        if unknown:
            raise ValueError(f"Parameters {sorted(unknown)} are not bound by the plan.")
        fixed = set(self.fixed_ids) & set(supplied)
        if fixed:
            raise ValueError(
                f"Parameters {sorted(fixed)} change the prepared structure; prepare "
                "the coupled problem again with their new values."
            )
        missing = declared - set(self.fixed_ids) - set(supplied)
        if missing:
            raise ValueError(
                f"Parameters {sorted(missing)} must be bound at every solve; refresh "
                "values are never retained by the prepared problem."
            )
        values: dict[str, ParameterValue] = dict(
            zip(self.fixed_ids, self.fixed_values, strict=True)
        )
        for binding in self.bindings:
            if binding.binding_id in supplied:
                values[binding.binding_id] = _checked_value(
                    binding, supplied[binding.binding_id]
                )
        return values

    def derivative_capability(
        self,
        owner_id: str,
        linear: bool,
        policy: LinearSolvePolicy | None,
        /,
    ) -> OwnerDerivativeCapability:
        """Parameters whose derivatives one coupled solve admits under ``policy``."""
        mode: DifferentiationMode = (
            "mathematical" if policy is None else policy.differentiation.mode
        )
        admitted: dict[str, DerivativeSurface] = {}
        refused = dict(_FIXED_STRUCTURE)
        for binding in self.bindings:
            reason = _refusal(binding, linear, mode)
            if reason is None and binding.derivative is not None:
                admitted[binding.binding_id] = binding.derivative
            else:
                refused[binding.binding_id] = reason or "no derivative surface declared"
        return OwnerDerivativeCapability(
            owner_id,
            admitted=admitted,
            refused=refused,
            route=DerivativeRoute.IMPLICIT if admitted else DerivativeRoute.STOPPED,
            conditions=_CONDITIONS if admitted else (),
        )


def _refusal(
    binding: ParameterBinding, linear: bool, mode: DifferentiationMode, /
) -> str | None:
    if binding.change == "reprepare":
        return "a change requires re-preparation; the value is fixed prepared structure"
    if binding.derivative is None:
        return "no derivative surface declared"
    if not linear:
        return (
            "the nonlinear coupled solve publishes no qualified implicit solution-map "
            "derivative"
        )
    match mode:
        case "mathematical":
            return None
        case "rhs-only":
            if binding.derivative is DerivativeSurface.PHYSICAL_PARAMETER:
                return "operator arrays are fixed under differentiation mode 'rhs-only'"
            return None
        case "algorithmic":
            return (
                "an unrolled algorithmic derivative is not the solution-map derivative "
                "of the coupled problem"
            )
        case "none":
            return "differentiation mode 'none' stops the coupled solve"
        case _:
            raise ValueError(f"Unknown differentiation mode {mode!r}.")


def prepare_parameters(
    bindings: tuple[ParameterBinding, ...],
    names: tuple[str, ...],
    values: Mapping[str, object] | None,
    /,
) -> tuple[PreparedParameters, dict[str, ParameterValue]]:
    """Validate parameter targets and reference values once at preparation.

    Returns the prepared parameters (with ``"reprepare"`` values fixed) and the
    checked reference values of every binding for preparation-time evaluation.
    """
    claimed: dict[tuple[str, str], str] = {}
    for binding in bindings:
        for target in binding.targets:
            key = (target.component, target.name)
            if key in claimed:
                raise ValueError(
                    f"Parameters {claimed[key]!r} and {binding.binding_id!r} both bind "
                    f"runtime input {target.name!r} of component {target.component!r}."
                )
            if target.component not in names:
                raise ValueError(
                    f"Parameter {binding.binding_id!r} names unknown component "
                    f"{target.component!r}."
                )
            claimed[key] = binding.binding_id
    supplied = {} if values is None else dict(values)
    declared = {binding.binding_id for binding in bindings}
    if set(supplied) != declared:
        raise ValueError(
            "Preparation binds a reference value of every declared parameter; "
            f"missing {sorted(declared - set(supplied))}, unknown "
            f"{sorted(set(supplied) - declared)}."
        )
    checked = {
        binding.binding_id: _checked_value(binding, supplied[binding.binding_id])
        for binding in bindings
    }
    fixed = tuple(binding for binding in bindings if binding.change == "reprepare")
    fixed_values = tuple(checked[binding.binding_id] for binding in fixed)
    if not all(isinstance(value, Array) for value in fixed_values):
        raise TypeError("Parameters fixed at preparation are arrays.")
    prepared = PreparedParameters(
        bindings=bindings,
        fixed_ids=tuple(binding.binding_id for binding in fixed),
        fixed_values=tuple(
            jax.lax.stop_gradient(value)
            for value in fixed_values
            if isinstance(value, Array)
        ),
    )
    return prepared, checked


__all__ = [
    "CoupledArguments",
    "ParameterBinding",
    "ParameterChange",
    "ParameterRole",
    "PreparedParameters",
    "RuntimeInput",
    "prepare_parameters",
]
