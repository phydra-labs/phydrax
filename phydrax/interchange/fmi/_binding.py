#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Physical coupling ports bound to FMI2 Co-Simulation communication variables.

An FMI2 Co-Simulation slave exchanges scalar values at communication points and
holds its inputs over each `fmi2DoStep` (no input derivatives are set). A
binding states, per variable, how that exchange realizes one physical coupling
port, and checks it against the model description: Real type, communication
causality and variability, and the declared `BaseUnit` dimension and factor.
Values cross the host boundary through exact unit factors only; affine units,
undeclared units, and waveforms are refused, never guessed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from enum import IntEnum
from fractions import Fraction
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ..._external_runtime import (
    _DERIVATIVE_FREE_ALTERNATIVES,
    _host_only,
    ExternalDerivativeSupport,
)
from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import ArraySpace
from ...solver.coupling import (
    AbstractHostCouplingParticipant,
    CouplingPort,
    CouplingSubsystemResult,
    CouplingWindow,
    HostParticipantState,
    HostRollback,
)
from ...typing import parse
from ...units import (
    AMOUNT,
    ANGLE,
    CURRENT,
    DimensionSignature,
    LENGTH,
    MASS,
    SI_REFERENCE_SYSTEM_ID,
    TEMPERATURE,
    TIME,
    UnitDefinition,
)
from ._session import (
    FMICoSimulationSession,
    FMIModelDescription,
    FMIState,
    FMIStepResult,
    FMIUnit,
    FMIVariable,
)


FMIPortRealization: TypeAlias = Literal["hold", "uniform-rate", "sample", "increment"]

_BASE_DIMENSIONS = {
    "kg": MASS,
    "m": LENGTH,
    "s": TIME,
    "A": CURRENT,
    "K": TEMPERATURE,
    "mol": AMOUNT,
    "rad": ANGLE,
}
_COMMUNICATION_VARIABILITY = ("continuous", "discrete")


class FMIWindowStatus(IntEnum):
    """Participant status of one FMI co-simulation window."""

    OK = 0
    WARNING = 1
    EARLY_RETURN = 2
    TERMINATED = 3
    NONFINITE_INPUT = 4


def _direction(realization: FMIPortRealization, /) -> tuple[str, str]:
    """Port direction and temporal kind one realization binds."""
    match realization:
        case "hold":
            return "input", "instantaneous"
        case "uniform-rate":
            return "input", "interval_integral"
        case "sample":
            return "output", "instantaneous"
        case "increment":
            return "output", "interval_integral"
        case _:
            raise ValueError(f"Unknown FMI port realization {realization!r}.")


class FMIVariableBinding(StrictModule, NonTrainableState):
    """Declared realization of one physical coupling port by one FMI2 Real variable.

    - `"hold"`: an instantaneous input port value is set at the communication
      point and held over the step.
    - `"uniform-rate"`: a whole-window input amount is delivered as the constant
      rate `amount / window` over the step, so exactly the amount enters.
    - `"sample"`: an instantaneous output port observes the variable at the
      reached communication point.
    - `"increment"`: a whole-window output amount is the change of a cumulative
      variable over the step.

    The port must be physically typed, scalar (one real coordinate), and
    endpoint-valued; its direction and temporal kind must match the realization.
    """

    port: CouplingPort
    variable: str = eqx.field(static=True)
    realization: FMIPortRealization = eqx.field(static=True)

    def __init__(
        self, variable: str, port: CouplingPort, realization: FMIPortRealization, /
    ) -> None:
        if not isinstance(variable, str) or not variable:
            raise ValueError("FMI variable binding requires a variable name.")
        if not isinstance(port, CouplingPort):
            raise TypeError("port must be a CouplingPort.")
        realization = parse(realization, FMIPortRealization, "realization")
        direction, temporal_kind = _direction(realization)
        if port.direction != direction or port.temporal_kind != temporal_kind:
            raise ValueError(
                f"FMI realization {realization!r} binds {temporal_kind} {direction} "
                f"ports; port {port.port_id!r} is {port.temporal_kind} {port.direction}."
            )
        if port.quantity is None:
            raise ValueError(
                f"Port {port.port_id!r} must declare its physical quantity to bind an "
                "FMI variable."
            )
        if port.waveform_plan is not None:
            raise ValueError(
                "FMI2 Co-Simulation exchanges communication-point values, not waveforms."
            )
        space = port.space
        if (
            not isinstance(space, ArraySpace)
            or math.prod(space.shape) != 1
            or not np.issubdtype(space.dtype, np.floating)
        ):
            raise ValueError(
                f"Port {port.port_id!r} must hold one real coordinate for one FMI "
                "scalar variable."
            )
        self.port = port
        self.variable = variable
        self.realization = realization


def fmi_unit_definition(unit: FMIUnit, /) -> UnitDefinition:
    """Exact multiplicative SI unit of one FMI `BaseUnit` declaration.

    A unit without `BaseUnit` states no dimension, an offset makes it affine, and
    candela has no Phydrax dimension axis; each is refused.
    """
    if not isinstance(unit, FMIUnit):
        raise TypeError("unit must be an FMIUnit.")
    if not unit.declared_base:
        raise ValueError(
            f"FMU unit {unit.name!r} declares no BaseUnit; the model description "
            "states no dimension for it."
        )
    if Fraction(unit.offset) != 0:
        raise ValueError(
            f"FMU unit {unit.name!r} is affine (offset {unit.offset}); physical "
            "coupling converts multiplicative units only."
        )
    factor = Fraction(unit.factor)
    if factor <= 0:
        raise ValueError(f"FMU unit {unit.name!r} requires a positive BaseUnit factor.")
    dimension = DimensionSignature()
    for base, exponent in unit.base_exponents:
        if base not in _BASE_DIMENSIONS:
            raise ValueError(
                f"FMU unit {unit.name!r} uses BaseUnit {base!r}, which has no "
                "Phydrax dimension axis."
            )
        dimension = dimension * _BASE_DIMENSIONS[base].power(exponent)
    return UnitDefinition(unit.name, dimension, SI_REFERENCE_SYSTEM_ID, factor)


def _variable_unit(
    model: FMIModelDescription, variable: FMIVariable, /
) -> UnitDefinition:
    if not variable.unit:
        raise ValueError(
            f"FMU variable {variable.name!r} declares no unit; its physical quantity "
            "cannot be established from the model description."
        )
    try:
        declared = model.unit(variable.unit)
    except KeyError as error:
        raise ValueError(
            f"FMU variable {variable.name!r} names unit {variable.unit!r}, which "
            "UnitDefinitions does not declare."
        ) from error
    return fmi_unit_definition(declared)


def _time_unit(model: FMIModelDescription, time_unit: UnitDefinition, /) -> None:
    """Require the coupling time coordinate to be the FMU's independent variable.

    FMI 2.0 defines an undeclared independent variable as `time` in seconds. A
    declared independent variable without a unit states no time unit, so no
    coupling time unit can be verified against it.
    """
    if not isinstance(time_unit, UnitDefinition):
        raise TypeError("time_unit must be a UnitDefinition.")
    if (
        time_unit.dimension != TIME
        or time_unit.reference_system_id != SI_REFERENCE_SYSTEM_ID
    ):
        raise ValueError("time_unit must be an SI unit of time.")
    independent = [
        variable for variable in model.variables if variable.causality == "independent"
    ]
    if len(independent) > 1:
        raise ValueError("An FMI 2.0 model declares at most one independent variable.")
    if not independent:
        if time_unit.scale_to_reference != 1:
            raise ValueError(
                "The FMU declares no independent variable, so its time is in seconds "
                f"(FMI 2.0), not the coupling time unit {time_unit.symbol!r}."
            )
        return
    variable = independent[0]
    if not variable.unit:
        raise ValueError(
            f"The FMU independent variable {variable.name!r} declares no unit; the "
            "coupling time unit cannot be verified against it."
        )
    declared = _variable_unit(model, variable)
    if declared.dimension != TIME or (
        declared.scale_to_reference != time_unit.scale_to_reference
    ):
        raise ValueError(
            f"The FMU independent variable is in {variable.unit!r}, not the "
            f"coupling time unit {time_unit.symbol!r}."
        )


def _checked_variable(
    model: FMIModelDescription, binding: FMIVariableBinding, /
) -> FMIVariable:
    try:
        variable = model.variable(binding.variable)
    except KeyError as error:
        raise ValueError(
            f"FMU {model.model_name!r} declares no variable {binding.variable!r}."
        ) from error
    if variable.type != "Real":
        raise ValueError(
            f"FMU variable {variable.name!r} is {variable.type}; physical ports bind "
            "Real variables."
        )
    causality = binding.port.direction
    if variable.causality != causality:
        raise ValueError(
            f"FMU variable {variable.name!r} has causality {variable.causality!r}; "
            f"realization {binding.realization!r} requires {causality!r}."
        )
    if variable.variability not in _COMMUNICATION_VARIABILITY:
        raise ValueError(
            f"FMU variable {variable.name!r} has variability {variable.variability!r} "
            "and cannot change at communication points."
        )
    return variable


def _port_factor(
    model: FMIModelDescription,
    binding: FMIVariableBinding,
    variable: FMIVariable,
    time_unit: UnitDefinition,
    /,
) -> tuple[UnitDefinition, Fraction]:
    """FMU unit and the exact factor converting port coordinates to FMU values.

    Inputs multiply the port coordinate by the factor (a uniform rate also divides
    by the window); outputs divide the FMU value by it.
    """
    quantity = binding.port.quantity
    if quantity is None:
        raise RuntimeError("A bound FMI port lost its physical quantity.")
    fmu_unit = _variable_unit(model, variable)
    port_unit = quantity.unit
    if port_unit.reference_system_id != SI_REFERENCE_SYSTEM_ID:
        raise ValueError(
            f"Port {binding.port.port_id!r} is not in the SI reference system of FMI "
            "BaseUnit declarations."
        )
    expected = port_unit.dimension
    scale = port_unit.scale_to_reference / fmu_unit.scale_to_reference
    if binding.realization == "uniform-rate":
        expected = expected / TIME
        scale = scale / time_unit.scale_to_reference
    if fmu_unit.dimension != expected:
        raise ValueError(
            f"FMU variable {variable.name!r} in {variable.unit!r} does not realize "
            f"port {binding.port.port_id!r} ({port_unit.symbol}) as "
            f"{binding.realization!r}."
        )
    return fmu_unit, scale


class FMICouplingBinding(StrictModule, NonTrainableState):
    """Checked physical port bindings of one FMI2 Co-Simulation model description.

    `time_unit` is the unit of the coupling time coordinate, which is the FMU's
    independent variable. `factors[k]` converts the port coordinate of
    `variables[k]` to the FMU value exactly. `rollback` is `"restore"` only when
    the FMU advertises `canGetAndSetFMUstate`.
    """

    variables: tuple[FMIVariableBinding, ...]
    time_unit: UnitDefinition
    fmu_units: tuple[UnitDefinition, ...]
    factors: tuple[float, ...] = eqx.field(static=True)
    value_references: tuple[int, ...] = eqx.field(static=True)
    model_name: str = eqx.field(static=True)
    guid: str = eqx.field(static=True)
    archive_sha256: str = eqx.field(static=True)
    rollback: HostRollback = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)

    def __init__(
        self,
        model: FMIModelDescription,
        variables: Sequence[FMIVariableBinding],
        /,
        *,
        time_unit: UnitDefinition,
    ) -> None:
        if not isinstance(model, FMIModelDescription):
            raise TypeError("model must be an FMIModelDescription.")
        bindings = tuple(variables)
        if not bindings or any(
            not isinstance(value, FMIVariableBinding) for value in bindings
        ):
            raise TypeError("variables must be a non-empty FMIVariableBinding sequence.")
        names = [binding.variable for binding in bindings]
        ports = [binding.port.port_id for binding in bindings]
        if len(set(names)) != len(names) or len(set(ports)) != len(ports):
            raise ValueError("Each FMU variable and each port is bound at most once.")
        _time_unit(model, time_unit)
        checked = tuple(_checked_variable(model, binding) for binding in bindings)
        resolved = tuple(
            _port_factor(model, binding, variable, time_unit)
            for binding, variable in zip(bindings, checked, strict=True)
        )
        self.variables = bindings
        self.time_unit = time_unit
        self.fmu_units = tuple(unit for unit, _ in resolved)
        self.factors = tuple(float(factor) for _, factor in resolved)
        self.value_references = tuple(variable.value_reference for variable in checked)
        self.model_name = model.model_name
        self.guid = model.guid
        self.archive_sha256 = model.archive_sha256
        self.rollback = "restore" if model.can_get_set_state else "none"
        self.binding_id = canonical_fingerprint(
            {
                "kind": "fmi-coupling-binding",
                "archive": model.archive_sha256,
                "guid": model.guid,
                "time_unit": time_unit.unit_id,
                "variables": [
                    {
                        "variable": binding.variable,
                        "value_reference": variable.value_reference,
                        "realization": binding.realization,
                        "port": binding.port.port_id,
                        "quantity": None
                        if binding.port.quantity is None
                        else binding.port.quantity.quantity_id,
                        "measurement": None
                        if binding.port.measurement is None
                        else binding.port.measurement.measurement_id,
                        "fmu_unit": unit.unit_id,
                        "factor": [factor.numerator, factor.denominator],
                    }
                    for binding, variable, (unit, factor) in zip(
                        bindings, checked, resolved, strict=True
                    )
                ],
            }
        )


def _port_value(port: CouplingPort, value: float, /) -> Any:
    space = port.space
    if not isinstance(space, ArraySpace):
        raise RuntimeError("A bound FMI port lost its scalar array space.")
    return jnp.full(space.shape, value, dtype=space.dtype)


def _coordinate(port: CouplingPort, value: Any, /) -> float:
    return float(np.asarray(port.space.flatten(port.space.validate(value)))[0])


def _step_status(step: FMIStepResult, /) -> FMIWindowStatus:
    """Window status of one step; a discarded step never counts as completed."""
    if step.terminated:
        return FMIWindowStatus.TERMINATED
    match step.status:
        case "ok":
            completed = FMIWindowStatus.OK
        case "warning":
            completed = FMIWindowStatus.WARNING
        case "discard":
            return FMIWindowStatus.EARLY_RETURN
        case _:
            raise RuntimeError(f"Unknown FMI step status {step.status!r}.")
    return FMIWindowStatus.EARLY_RETURN if step.early_return else completed


class FMICouplingParticipant(AbstractHostCouplingParticipant):
    """One live FMI2 Co-Simulation session bound as a host coupling participant.

    Ports follow the binding order: inputs from `"hold"` and `"uniform-rate"`
    variables, outputs from `"sample"` and `"increment"` variables. Each window
    sets every bound input at the accepted communication point and performs one
    `fmi2DoStep` to the window end. An early return, a termination request, or a
    non-finite input (which is refused before any process I/O) fails the window.
    State capture and restore use the FMU's native get/set state; the participant
    offers no adjoint.
    """

    session: FMICoSimulationSession
    binding: FMICouplingBinding
    input_ports: tuple[CouplingPort, ...]
    output_ports: tuple[CouplingPort, ...]
    rollback: HostRollback = eqx.field(static=True)
    subsystem_id: str = eqx.field(static=True)

    def __init__(
        self,
        session: FMICoSimulationSession,
        binding: FMICouplingBinding,
        /,
        *,
        subsystem_id: str,
    ) -> None:
        if not isinstance(session, FMICoSimulationSession):
            raise TypeError("session must be an FMICoSimulationSession.")
        if not isinstance(binding, FMICouplingBinding):
            raise TypeError("binding must be an FMICouplingBinding.")
        if (
            session.model.archive_sha256 != binding.archive_sha256
            or session.model.guid != binding.guid
        ):
            raise ValueError("The FMI binding was checked against a different FMU.")
        if session.closed or session.terminated:
            raise ValueError("An FMI coupling participant requires an active session.")
        if not isinstance(subsystem_id, str) or not subsystem_id:
            raise ValueError("subsystem_id must be a non-empty string.")
        self.session = session
        self.binding = binding
        self.input_ports = tuple(
            value.port for value in binding.variables if value.port.direction == "input"
        )
        self.output_ports = tuple(
            value.port for value in binding.variables if value.port.direction == "output"
        )
        self.rollback = binding.rollback
        self.subsystem_id = subsystem_id

    @property
    def derivative_support(self) -> ExternalDerivativeSupport:
        """A forward co-simulation offers no adjoint; none is selected for it."""
        return ExternalDerivativeSupport("none", _DERIVATIVE_FREE_ALTERNATIVES)

    @property
    def time_unit(self) -> UnitDefinition:
        """The checked unit of the FMU independent variable and coupling clock."""
        return self.binding.time_unit

    def communication_time(self) -> float:
        return self.session.time

    def capture(self) -> FMIState:
        return self.session.save_state()

    def restore(self, checkpoint: Any, /) -> None:
        if not isinstance(checkpoint, FMIState):
            raise TypeError("FMI checkpoints are FMIState tokens.")
        self.session.restore_state(checkpoint)

    def release(self, checkpoint: Any, /) -> None:
        if not isinstance(checkpoint, FMIState):
            raise TypeError("FMI checkpoints are FMIState tokens.")
        self.session.free_state(checkpoint)

    def evidence_id(self) -> str:
        return self.session.artifact.artifact_id

    def _outputs(
        self, after: Mapping[str, Any], before: Mapping[str, Any] | None, /
    ) -> tuple[Any, ...]:
        """Output-port values from observations; increments need a start sample."""
        outputs: list[Any] = []
        for binding, factor in zip(
            self.binding.variables, self.binding.factors, strict=True
        ):
            match binding.realization:
                case "sample":
                    value = after[binding.variable] / factor
                case "increment":
                    value = (
                        0.0
                        if before is None
                        else (after[binding.variable] - before[binding.variable]) / factor
                    )
                case "hold" | "uniform-rate":
                    continue
                case _:
                    raise ValueError(
                        f"Unknown FMI port realization {binding.realization!r}."
                    )
            outputs.append(_port_value(binding.port, value))
        return tuple(outputs)

    def _observed(self, realizations: tuple[str, ...], /) -> dict[str, Any]:
        names = tuple(
            binding.variable
            for binding in self.binding.variables
            if binding.realization in realizations
        )
        return self.session.get_values(names) if names else {}

    def _input_values(
        self, inputs: tuple[Any, ...], size: float, /
    ) -> dict[str, float] | None:
        """FMU input values, or None when a non-finite input must be refused."""
        values: dict[str, float] = {}
        bound = (
            (binding, factor)
            for binding, factor in zip(
                self.binding.variables, self.binding.factors, strict=True
            )
            if binding.port.direction == "input"
        )
        for (binding, factor), value in zip(bound, inputs, strict=True):
            coordinate = _coordinate(binding.port, value) * factor
            if binding.realization == "uniform-rate":
                coordinate /= size
            if not math.isfinite(coordinate):
                return None
            values[binding.variable] = coordinate
        return values

    def initial_outputs(self) -> tuple[Any, ...]:
        _host_only()
        return self._outputs(self._observed(("sample",)), None)

    def advance_window(
        self,
        window: CouplingWindow,
        start_state: Any,
        inputs: tuple[Any, ...],
        args: Any,
        /,
    ) -> CouplingSubsystemResult:
        """Set the bound inputs, step the live FMU once, and observe its outputs."""
        del args
        _host_only(window, start_state, inputs)
        if not isinstance(start_state, HostParticipantState):
            raise TypeError("FMI participant state must be HostParticipantState.")
        start, end = float(window.start), float(window.end)
        live = self.session.time
        if not math.isclose(live, start, rel_tol=1e-12, abs_tol=1e-12):
            raise RuntimeError(
                f"FMU participant {self.subsystem_id!r} is at {live!r}, not the window "
                f"start {start!r}; restore the accepted checkpoint first."
            )
        values = self._input_values(tuple(inputs), end - start)
        if values is None:
            # The process is not advanced: it stays exactly at the accepted state.
            return self._result(
                start_state,
                live,
                self._outputs(self._observed(("sample",)), None),
                FMIWindowStatus.NONFINITE_INPUT,
                0,
            )
        before = self._observed(("increment",))
        self.session.set_values(values)
        step = self.session.advance(end)
        after = self._observed(("sample", "increment"))
        return self._result(
            start_state,
            step.reached_time,
            self._outputs(after, before),
            _step_status(step),
            1,
        )

    def _result(
        self,
        start_state: HostParticipantState,
        reached: float,
        outputs: tuple[Any, ...],
        status: FMIWindowStatus,
        work: int,
        /,
    ) -> CouplingSubsystemResult:
        candidate = HostParticipantState(
            jnp.asarray(reached, dtype=start_state.time.dtype),
            start_state.accepted_windows + work,
        )
        return CouplingSubsystemResult(
            candidate,
            outputs,
            successful=status in (FMIWindowStatus.OK, FMIWindowStatus.WARNING),
            status=int(status),
            work=work,
        )


__all__ = [
    "FMICouplingBinding",
    "FMICouplingParticipant",
    "FMIPortRealization",
    "FMIVariableBinding",
    "FMIWindowStatus",
    "fmi_unit_definition",
]
