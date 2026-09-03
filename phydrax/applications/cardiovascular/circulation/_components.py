#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import StrEnum
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ....dynamics import (
    DAEComponent,
    DAEDerivativeIncidence,
    DAEEquationBlock,
    DAEPort,
    DAEVariableBlock,
)


PressureWaveform = Callable[[Array], Array]
FlowWaveform = Callable[[Array], Array]
ElastanceWaveform = Callable[[Array], Array]
VolumeRateLaw = Callable[[Array, Any], Array]


class StorageOwner(StrEnum):
    """Exclusive owner of a component's volume state."""

    NONE = "none"
    CIRCULATION = "circulation"
    MECHANICS = "mechanics"


class _ConstantWaveform(StrictModule):
    value: Array

    def __call__(self, time: Array, /) -> Array:
        del time
        return self.value


class PeriodicElastance(StrictModule):
    """Smooth periodic raised-cosine elastance waveform in kPa/mm³."""

    minimum: Array
    maximum: Array
    cycle_length: Array
    systolic_duration: Array
    phase_offset: Array
    waveform_id: str = eqx.field(static=True)

    def __init__(
        self,
        minimum: ArrayLike,
        maximum: ArrayLike,
        cycle_length: ArrayLike,
        systolic_duration: ArrayLike,
        /,
        *,
        phase_offset: ArrayLike = 0.0,
    ) -> None:
        values = tuple(
            jnp.asarray(value)
            for value in (
                minimum,
                maximum,
                cycle_length,
                systolic_duration,
                phase_offset,
            )
        )
        if any(value.shape != () for value in values):
            raise ValueError("Elastance waveform parameters must be scalars.")
        host = tuple(float(value) for value in values)
        if (
            any(not np.isfinite(value) for value in host)
            or host[0] <= 0.0
            or host[1] < host[0]
            or host[2] <= 0.0
            or host[3] <= 0.0
            or host[3] > host[2]
        ):
            raise ValueError("Elastance parameters must be finite and physical.")
        (
            self.minimum,
            self.maximum,
            self.cycle_length,
            self.systolic_duration,
            self.phase_offset,
        ) = values
        self.waveform_id = canonical_fingerprint(
            {
                "kind": "periodic-elastance",
                "minimum": host[0].hex(),
                "maximum": host[1].hex(),
                "cycle_length": host[2].hex(),
                "systolic_duration": host[3].hex(),
                "phase_offset": host[4].hex(),
            }
        )

    def __call__(self, time: Array, /) -> Array:
        phase = jnp.mod(jnp.asarray(time) - self.phase_offset, self.cycle_length)
        normalized = jnp.clip(phase / self.systolic_duration, 0.0, 1.0)
        activation = 0.5 * (1.0 - jnp.cos(2.0 * jnp.pi * normalized))
        activation = jnp.where(phase <= self.systolic_duration, activation, 0.0)
        return self.minimum + (self.maximum - self.minimum) * activation


class PressureFlowComponent(StrictModule):
    """Typed pressure/flow wrapper around the canonical generic DAE component.

    Every hydraulic port contains one pressure potential in kPa and one directed
    volume flow in mm³/ms. Two-port components use ``inlet`` and ``outlet`` and
    positive flow from inlet to outlet.
    """

    __strict_abstract__ = True

    dae_component: DAEComponent
    initial_values: tuple[tuple[str, Array], ...]
    component_kind: str = eqx.field(static=True)
    storage_owner: StorageOwner = eqx.field(static=True)
    storage_variable_names: tuple[str, ...] = eqx.field(static=True)
    component_id: str = eqx.field(static=True)

    def __init__(
        self,
        dae_component: DAEComponent,
        /,
        *,
        component_kind: str,
        parameters: Sequence[tuple[str, float | str]],
        storage_owner: StorageOwner = StorageOwner.NONE,
        storage_variable_names: Sequence[str] = (),
        initial_values: Sequence[tuple[str, ArrayLike]] = (),
    ) -> None:
        if not isinstance(dae_component, DAEComponent):
            raise TypeError("dae_component must be a DAEComponent.")
        kind = str(component_kind).strip()
        if not kind:
            raise ValueError("component_kind must be non-empty.")
        if not isinstance(storage_owner, StorageOwner):
            raise TypeError("storage_owner must be a StorageOwner.")
        storage = tuple(str(value) for value in storage_variable_names)
        variable_names = {value.name for value in dae_component.variables}
        if len(set(storage)) != len(storage) or set(storage) - variable_names:
            raise ValueError("Storage variables must be unique component variables.")
        if storage_owner is StorageOwner.NONE and storage:
            raise ValueError("Storage variables require an explicit exclusive owner.")
        if (
            storage_owner is not StorageOwner.NONE
            and not storage
            and storage_owner is not StorageOwner.MECHANICS
        ):
            raise ValueError("Circulation-owned storage requires a volume variable.")
        initial = tuple((str(name), jnp.asarray(value)) for name, value in initial_values)
        if len({name for name, _ in initial}) != len(initial):
            raise ValueError("Initial component values must have unique names.")
        if {name for name, _ in initial} - variable_names:
            raise ValueError("Initial values reference unknown component variables.")
        if any(
            value.shape != () or not bool(jnp.isfinite(value)) for _, value in initial
        ):
            raise ValueError("Initial component values must be finite scalars.")
        self.dae_component = dae_component
        self.initial_values = initial
        self.component_kind = kind
        self.storage_owner = storage_owner
        self.storage_variable_names = storage
        self.component_id = canonical_fingerprint(
            {
                "kind": kind,
                "name": dae_component.name,
                "ports": [value.name for value in dae_component.ports],
                "storage_owner": storage_owner.value,
                "storage": list(storage),
                "parameters": [list(value) for value in parameters],
            }
        )

    @property
    def name(self) -> str:
        return self.dae_component.name

    @property
    def ports(self) -> tuple[DAEPort, ...]:
        return self.dae_component.ports

    def port(self, name: str, /) -> DAEPort:
        for value in self.dae_component.ports:
            if value.name == name:
                return value
        raise KeyError(f"Unknown pressure/flow port {name!r} on {self.name!r}.")

    def port_id(self, name: str, /) -> str:
        self.port(name)
        return f"{self.name}.{name}"

    def initial_value(self, variable_name: str, /) -> Array:
        for name, value in self.initial_values:
            if name == variable_name:
                return value
        return jnp.asarray(0.0)


class _ConservationResidual(StrictModule):
    def __call__(self, time: Array, jet, args: Any, /) -> Array:
        del time, args
        return jet.value("flow_in") - jet.value("flow_out")


class _PressureEqualityResidual(StrictModule):
    def __call__(self, time: Array, jet, args: Any, /) -> Array:
        del time, args
        return jet.value("pressure_in") - jet.value("pressure_out")


class _ResistanceResidual(StrictModule):
    resistance: Array

    def __call__(self, time: Array, jet, args: Any, /) -> Array:
        del time, args
        return (
            jet.value("pressure_in")
            - jet.value("pressure_out")
            - self.resistance * jet.value("flow_out")
        )


class _ComplianceConstitutiveResidual(StrictModule):
    compliance: Array
    unstressed_volume: Array
    reference_pressure: Array

    def __call__(self, time: Array, jet, args: Any, /) -> Array:
        del time, args
        return (
            jet.value("volume")
            - self.unstressed_volume
            - self.compliance * (jet.value("pressure_in") - self.reference_pressure)
        )


class _VolumeBalanceResidual(StrictModule):
    def __call__(self, time: Array, jet, args: Any, /) -> Array:
        del time, args
        return jet.value("volume", 1) - jet.value("flow_in") + jet.value("flow_out")


class _InertanceResidual(StrictModule):
    inertance: Array

    def __call__(self, time: Array, jet, args: Any, /) -> Array:
        del time, args
        return (
            jet.value("pressure_in")
            - jet.value("pressure_out")
            - self.inertance * jet.value("flow_out", 1)
        )


class _RCRProximalResidual(StrictModule):
    resistance: Array

    def __call__(self, time: Array, jet, args: Any, /) -> Array:
        del time, args
        return (
            jet.value("pressure_in")
            - jet.value("pressure_capacitor")
            - self.resistance * jet.value("flow_in")
        )


class _RCRDistalResidual(StrictModule):
    resistance: Array

    def __call__(self, time: Array, jet, args: Any, /) -> Array:
        del time, args
        return (
            jet.value("pressure_capacitor")
            - jet.value("pressure_out")
            - self.resistance * jet.value("flow_out")
        )


class _RCRConstitutiveResidual(StrictModule):
    compliance: Array
    unstressed_volume: Array
    reference_pressure: Array

    def __call__(self, time: Array, jet, args: Any, /) -> Array:
        del time, args
        return (
            jet.value("volume")
            - self.unstressed_volume
            - self.compliance
            * (jet.value("pressure_capacitor") - self.reference_pressure)
        )


class _PrescribedPressureResidual(StrictModule):
    waveform: PressureWaveform

    def __call__(self, time: Array, jet, args: Any, /) -> Array:
        del args
        return jet.value("pressure_out") - jet.value("pressure_in") - self.waveform(time)


class _PrescribedFlowResidual(StrictModule):
    waveform: FlowWaveform

    def __call__(self, time: Array, jet, args: Any, /) -> Array:
        del args
        return jet.value("flow_out") - self.waveform(time)


class _ElastanceResidual(StrictModule):
    elastance: ElastanceWaveform
    unstressed_volume: Array
    reference_pressure: Array

    def __call__(self, time: Array, jet, args: Any, /) -> Array:
        del args
        return (
            jet.value("pressure_in")
            - self.reference_pressure
            - self.elastance(time) * (jet.value("volume") - self.unstressed_volume)
        )


class _MechanicsVolumeRateResidual(StrictModule):
    volume_rate: VolumeRateLaw

    def __call__(self, time: Array, jet, args: Any, /) -> Array:
        return jet.value("flow_in") - jet.value("flow_out") - self.volume_rate(time, args)


def _positive_scalar(value: ArrayLike, owner: str, /) -> tuple[Array, float]:
    resolved = jnp.asarray(value)
    if resolved.shape != ():
        raise ValueError(f"{owner} must be a scalar.")
    host = float(resolved)
    if not np.isfinite(host) or host <= 0.0:
        raise ValueError(f"{owner} must be positive and finite.")
    return resolved, host


def _finite_scalar(value: ArrayLike, owner: str, /) -> tuple[Array, float]:
    resolved = jnp.asarray(value)
    if resolved.shape != ():
        raise ValueError(f"{owner} must be a scalar.")
    host = float(resolved)
    if not np.isfinite(host):
        raise ValueError(f"{owner} must be finite.")
    return resolved, host


def _two_port_variables(
    pressure_scale: float,
    flow_scale: float,
    /,
    *,
    flow_derivative: bool = False,
) -> tuple[DAEVariableBlock, ...]:
    order = 1 if flow_derivative else 0
    return (
        DAEVariableBlock("pressure_in", (), 0, pressure_scale),
        DAEVariableBlock("pressure_out", (), 0, pressure_scale),
        DAEVariableBlock("flow_in", (), order, flow_scale),
        DAEVariableBlock("flow_out", (), order, flow_scale),
    )


def _two_ports() -> tuple[DAEPort, DAEPort]:
    return (
        DAEPort("inlet", ("pressure_in",), ("flow_in",)),
        DAEPort("outlet", ("pressure_out",), ("flow_out",)),
    )


def _incidence(*entries: tuple[str, int]) -> tuple[DAEDerivativeIncidence, ...]:
    return tuple(DAEDerivativeIncidence(name, order) for name, order in entries)


class Resistance(PressureFlowComponent):
    """Passive two-port hydraulic resistance, Δp = Rq."""

    resistance: Array

    def __init__(
        self,
        name: str,
        resistance: ArrayLike,
        /,
        *,
        pressure_scale: float = 10.0,
        flow_scale: float = 1.0,
    ) -> None:
        resistance_, host = _positive_scalar(resistance, "resistance")
        p_scale = _positive_scalar(pressure_scale, "pressure_scale")[1]
        q_scale = _positive_scalar(flow_scale, "flow_scale")[1]
        component = DAEComponent(
            name,
            _two_port_variables(p_scale, q_scale),
            (
                DAEEquationBlock(
                    "conserve_flow",
                    _ConservationResidual(),
                    _incidence(("flow_in", 0), ("flow_out", 0)),
                ),
                DAEEquationBlock(
                    "resistance",
                    _ResistanceResidual(resistance_),
                    _incidence(
                        ("pressure_in", 0),
                        ("pressure_out", 0),
                        ("flow_out", 0),
                    ),
                ),
            ),
            _two_ports(),
        )
        PressureFlowComponent.__init__(
            self,
            component,
            component_kind="hydraulic-resistance",
            parameters=(("resistance_kPa_ms_per_mm3", host),),
        )
        self.resistance = resistance_


class Compliance(PressureFlowComponent):
    """Two-port compliant reservoir with circulation-owned volume storage."""

    compliance: Array
    unstressed_volume: Array
    reference_pressure: Array
    initial_volume: Array

    def __init__(
        self,
        name: str,
        compliance: ArrayLike,
        /,
        *,
        unstressed_volume: ArrayLike = 0.0,
        reference_pressure: ArrayLike = 0.0,
        initial_volume: ArrayLike | None = None,
        pressure_scale: float = 10.0,
        flow_scale: float = 1.0,
        volume_scale: float = 100.0,
    ) -> None:
        compliance_, compliance_host = _positive_scalar(compliance, "compliance")
        unstressed, unstressed_host = _finite_scalar(
            unstressed_volume, "unstressed_volume"
        )
        reference, reference_host = _finite_scalar(
            reference_pressure, "reference_pressure"
        )
        initial = unstressed if initial_volume is None else jnp.asarray(initial_volume)
        initial, initial_host = _finite_scalar(initial, "initial_volume")
        if initial_host < 0.0:
            raise ValueError("initial_volume must be nonnegative.")
        initial_pressure = reference + (initial - unstressed) / compliance_
        p_scale = _positive_scalar(pressure_scale, "pressure_scale")[1]
        q_scale = _positive_scalar(flow_scale, "flow_scale")[1]
        v_scale = _positive_scalar(volume_scale, "volume_scale")[1]
        variables = _two_port_variables(p_scale, q_scale) + (
            DAEVariableBlock("volume", (), 1, v_scale),
        )
        equations = (
            DAEEquationBlock(
                "equal_pressure",
                _PressureEqualityResidual(),
                _incidence(("pressure_in", 0), ("pressure_out", 0)),
            ),
            DAEEquationBlock(
                "compliance",
                _ComplianceConstitutiveResidual(compliance_, unstressed, reference),
                _incidence(("volume", 0), ("pressure_in", 0)),
            ),
            DAEEquationBlock(
                "volume_balance",
                _VolumeBalanceResidual(),
                _incidence(("volume", 1), ("flow_in", 0), ("flow_out", 0)),
            ),
        )
        component = DAEComponent(name, variables, equations, _two_ports())
        PressureFlowComponent.__init__(
            self,
            component,
            component_kind="hydraulic-compliance",
            parameters=(
                ("compliance_mm3_per_kPa", compliance_host),
                ("unstressed_volume_mm3", unstressed_host),
                ("reference_pressure_kPa", reference_host),
            ),
            storage_owner=StorageOwner.CIRCULATION,
            storage_variable_names=("volume",),
            initial_values=(
                ("pressure_in", initial_pressure),
                ("pressure_out", initial_pressure),
                ("volume", initial),
            ),
        )
        self.compliance = compliance_
        self.unstressed_volume = unstressed
        self.reference_pressure = reference
        self.initial_volume = initial

    def stored_energy(self, volume: ArrayLike, /) -> Array:
        displacement = jnp.asarray(volume) - self.unstressed_volume
        return 0.5 * displacement * displacement / self.compliance


class Inertance(PressureFlowComponent):
    """Passive blood inertance, Δp = L dq/dt."""

    inertance: Array

    def __init__(
        self,
        name: str,
        inertance: ArrayLike,
        /,
        *,
        pressure_scale: float = 10.0,
        flow_scale: float = 1.0,
    ) -> None:
        inertance_, host = _positive_scalar(inertance, "inertance")
        p_scale = _positive_scalar(pressure_scale, "pressure_scale")[1]
        q_scale = _positive_scalar(flow_scale, "flow_scale")[1]
        component = DAEComponent(
            name,
            _two_port_variables(p_scale, q_scale, flow_derivative=True),
            (
                DAEEquationBlock(
                    "conserve_flow",
                    _ConservationResidual(),
                    _incidence(("flow_in", 0), ("flow_out", 0)),
                ),
                DAEEquationBlock(
                    "inertance",
                    _InertanceResidual(inertance_),
                    _incidence(
                        ("pressure_in", 0),
                        ("pressure_out", 0),
                        ("flow_out", 1),
                    ),
                ),
            ),
            _two_ports(),
        )
        PressureFlowComponent.__init__(
            self,
            component,
            component_kind="hydraulic-inertance",
            parameters=(("inertance_kPa_ms2_per_mm3", host),),
        )
        self.inertance = inertance_

    def stored_energy(self, flow: ArrayLike, /) -> Array:
        flow_ = jnp.asarray(flow)
        return 0.5 * self.inertance * flow_ * flow_


class WindkesselRCR(PressureFlowComponent):
    """Three-element proximal-resistance/compliance/distal-resistance model."""

    proximal_resistance: Array
    compliance: Array
    distal_resistance: Array
    unstressed_volume: Array
    reference_pressure: Array
    initial_volume: Array

    def __init__(
        self,
        name: str,
        proximal_resistance: ArrayLike,
        compliance: ArrayLike,
        distal_resistance: ArrayLike,
        /,
        *,
        unstressed_volume: ArrayLike = 0.0,
        reference_pressure: ArrayLike = 0.0,
        initial_volume: ArrayLike | None = None,
        pressure_scale: float = 10.0,
        flow_scale: float = 1.0,
        volume_scale: float = 100.0,
    ) -> None:
        proximal, proximal_host = _positive_scalar(
            proximal_resistance, "proximal_resistance"
        )
        compliance_, compliance_host = _positive_scalar(compliance, "compliance")
        distal, distal_host = _positive_scalar(distal_resistance, "distal_resistance")
        unstressed, unstressed_host = _finite_scalar(
            unstressed_volume, "unstressed_volume"
        )
        reference, reference_host = _finite_scalar(
            reference_pressure, "reference_pressure"
        )
        initial = unstressed if initial_volume is None else jnp.asarray(initial_volume)
        initial, initial_host = _finite_scalar(initial, "initial_volume")
        if initial_host < 0.0:
            raise ValueError("initial_volume must be nonnegative.")
        capacitor_pressure = reference + (initial - unstressed) / compliance_
        p_scale = _positive_scalar(pressure_scale, "pressure_scale")[1]
        q_scale = _positive_scalar(flow_scale, "flow_scale")[1]
        v_scale = _positive_scalar(volume_scale, "volume_scale")[1]
        variables = _two_port_variables(p_scale, q_scale) + (
            DAEVariableBlock("pressure_capacitor", (), 0, p_scale),
            DAEVariableBlock("volume", (), 1, v_scale),
        )
        equations = (
            DAEEquationBlock(
                "proximal_resistance",
                _RCRProximalResidual(proximal),
                _incidence(
                    ("pressure_in", 0),
                    ("pressure_capacitor", 0),
                    ("flow_in", 0),
                ),
            ),
            DAEEquationBlock(
                "distal_resistance",
                _RCRDistalResidual(distal),
                _incidence(
                    ("pressure_capacitor", 0),
                    ("pressure_out", 0),
                    ("flow_out", 0),
                ),
            ),
            DAEEquationBlock(
                "compliance",
                _RCRConstitutiveResidual(compliance_, unstressed, reference),
                _incidence(("volume", 0), ("pressure_capacitor", 0)),
            ),
            DAEEquationBlock(
                "volume_balance",
                _VolumeBalanceResidual(),
                _incidence(("volume", 1), ("flow_in", 0), ("flow_out", 0)),
            ),
        )
        component = DAEComponent(name, variables, equations, _two_ports())
        PressureFlowComponent.__init__(
            self,
            component,
            component_kind="windkessel-rcr",
            parameters=(
                ("proximal_resistance_kPa_ms_per_mm3", proximal_host),
                ("compliance_mm3_per_kPa", compliance_host),
                ("distal_resistance_kPa_ms_per_mm3", distal_host),
                ("unstressed_volume_mm3", unstressed_host),
                ("reference_pressure_kPa", reference_host),
            ),
            storage_owner=StorageOwner.CIRCULATION,
            storage_variable_names=("volume",),
            initial_values=(
                ("pressure_in", capacitor_pressure),
                ("pressure_out", capacitor_pressure),
                ("pressure_capacitor", capacitor_pressure),
                ("volume", initial),
            ),
        )
        self.proximal_resistance = proximal
        self.compliance = compliance_
        self.distal_resistance = distal
        self.unstressed_volume = unstressed
        self.reference_pressure = reference
        self.initial_volume = initial

    def stored_energy(self, volume: ArrayLike, /) -> Array:
        displacement = jnp.asarray(volume) - self.unstressed_volume
        return 0.5 * displacement * displacement / self.compliance


class PressureSource(PressureFlowComponent):
    """Ideal prescribed pressure-rise source."""

    waveform: PressureWaveform
    waveform_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        pressure: ArrayLike | PressureWaveform,
        /,
        *,
        waveform_id: str | None = None,
        pressure_scale: float = 10.0,
        flow_scale: float = 1.0,
    ) -> None:
        if callable(pressure):
            if waveform_id is None or not str(waveform_id).strip():
                raise ValueError(
                    "Callable pressure sources require a stable waveform_id."
                )
            waveform = pressure
            identifier = str(waveform_id).strip()
        else:
            value, host = _finite_scalar(pressure, "pressure")
            waveform = _ConstantWaveform(value)
            identifier = f"constant-pressure:{host.hex()}"
        p_scale = _positive_scalar(pressure_scale, "pressure_scale")[1]
        q_scale = _positive_scalar(flow_scale, "flow_scale")[1]
        component = DAEComponent(
            name,
            _two_port_variables(p_scale, q_scale),
            (
                DAEEquationBlock(
                    "conserve_flow",
                    _ConservationResidual(),
                    _incidence(("flow_in", 0), ("flow_out", 0)),
                ),
                DAEEquationBlock(
                    "prescribed_pressure",
                    _PrescribedPressureResidual(waveform),
                    _incidence(("pressure_in", 0), ("pressure_out", 0)),
                ),
            ),
            _two_ports(),
        )
        initial_pressure = waveform(jnp.asarray(0.0))
        PressureFlowComponent.__init__(
            self,
            component,
            component_kind="pressure-source",
            parameters=(("waveform_id", identifier),),
            initial_values=(("pressure_out", initial_pressure),),
        )
        self.waveform = waveform
        self.waveform_id = identifier


class FlowSource(PressureFlowComponent):
    """Ideal prescribed volume-flow source."""

    waveform: FlowWaveform
    waveform_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        flow: ArrayLike | FlowWaveform,
        /,
        *,
        waveform_id: str | None = None,
        pressure_scale: float = 10.0,
        flow_scale: float = 1.0,
    ) -> None:
        if callable(flow):
            if waveform_id is None or not str(waveform_id).strip():
                raise ValueError("Callable flow sources require a stable waveform_id.")
            waveform = flow
            identifier = str(waveform_id).strip()
        else:
            value, host = _finite_scalar(flow, "flow")
            waveform = _ConstantWaveform(value)
            identifier = f"constant-flow:{host.hex()}"
        p_scale = _positive_scalar(pressure_scale, "pressure_scale")[1]
        q_scale = _positive_scalar(flow_scale, "flow_scale")[1]
        component = DAEComponent(
            name,
            _two_port_variables(p_scale, q_scale),
            (
                DAEEquationBlock(
                    "conserve_flow",
                    _ConservationResidual(),
                    _incidence(("flow_in", 0), ("flow_out", 0)),
                ),
                DAEEquationBlock(
                    "prescribed_flow",
                    _PrescribedFlowResidual(waveform),
                    _incidence(
                        ("flow_out", 0),
                    ),
                ),
            ),
            _two_ports(),
        )
        initial_flow = waveform(jnp.asarray(0.0))
        PressureFlowComponent.__init__(
            self,
            component,
            component_kind="flow-source",
            parameters=(("waveform_id", identifier),),
            initial_values=(("flow_in", initial_flow), ("flow_out", initial_flow)),
        )
        self.waveform = waveform
        self.waveform_id = identifier


class TimeVaryingElastance(PressureFlowComponent):
    """Circulation-owned chamber with prescribed differentiable elastance."""

    elastance: ElastanceWaveform
    elastance_id: str = eqx.field(static=True)
    unstressed_volume: Array
    reference_pressure: Array
    initial_volume: Array

    def __init__(
        self,
        name: str,
        elastance: ElastanceWaveform,
        /,
        *,
        elastance_id: str | None = None,
        unstressed_volume: ArrayLike = 0.0,
        reference_pressure: ArrayLike = 0.0,
        initial_volume: ArrayLike,
        pressure_scale: float = 10.0,
        flow_scale: float = 1.0,
        volume_scale: float = 100.0,
    ) -> None:
        if not callable(elastance):
            raise TypeError("elastance must be callable.")
        if isinstance(elastance, PeriodicElastance):
            identifier = elastance.waveform_id
        elif elastance_id is not None and str(elastance_id).strip():
            identifier = str(elastance_id).strip()
        else:
            raise ValueError("Custom elastance callables require a stable elastance_id.")
        unstressed, unstressed_host = _finite_scalar(
            unstressed_volume, "unstressed_volume"
        )
        reference, reference_host = _finite_scalar(
            reference_pressure, "reference_pressure"
        )
        initial, initial_host = _finite_scalar(initial_volume, "initial_volume")
        if initial_host < 0.0:
            raise ValueError("initial_volume must be nonnegative.")
        initial_elastance = jnp.asarray(elastance(jnp.asarray(0.0)))
        if initial_elastance.shape != () or not bool(
            jnp.isfinite(initial_elastance) & (initial_elastance > 0.0)
        ):
            raise ValueError("elastance must return a finite positive scalar.")
        initial_pressure = reference + initial_elastance * (initial - unstressed)
        p_scale = _positive_scalar(pressure_scale, "pressure_scale")[1]
        q_scale = _positive_scalar(flow_scale, "flow_scale")[1]
        v_scale = _positive_scalar(volume_scale, "volume_scale")[1]
        variables = _two_port_variables(p_scale, q_scale) + (
            DAEVariableBlock("volume", (), 1, v_scale),
        )
        equations = (
            DAEEquationBlock(
                "equal_pressure",
                _PressureEqualityResidual(),
                _incidence(("pressure_in", 0), ("pressure_out", 0)),
            ),
            DAEEquationBlock(
                "elastance",
                _ElastanceResidual(elastance, unstressed, reference),
                _incidence(("pressure_in", 0), ("volume", 0)),
            ),
            DAEEquationBlock(
                "volume_balance",
                _VolumeBalanceResidual(),
                _incidence(("volume", 1), ("flow_in", 0), ("flow_out", 0)),
            ),
        )
        component = DAEComponent(name, variables, equations, _two_ports())
        PressureFlowComponent.__init__(
            self,
            component,
            component_kind="time-varying-elastance",
            parameters=(
                ("elastance_id", identifier),
                ("unstressed_volume_mm3", unstressed_host),
                ("reference_pressure_kPa", reference_host),
            ),
            storage_owner=StorageOwner.CIRCULATION,
            storage_variable_names=("volume",),
            initial_values=(
                ("pressure_in", initial_pressure),
                ("pressure_out", initial_pressure),
                ("volume", initial),
            ),
        )
        self.elastance = elastance
        self.elastance_id = identifier
        self.unstressed_volume = unstressed
        self.reference_pressure = reference
        self.initial_volume = initial

    def stored_energy(self, time: ArrayLike, volume: ArrayLike, /) -> Array:
        displacement = jnp.asarray(volume) - self.unstressed_volume
        return 0.5 * self.elastance(jnp.asarray(time)) * displacement * displacement


class MechanicsChamberCoupling(PressureFlowComponent):
    """Storage-free DAE adapter for a mechanics-owned chamber volume rate."""

    mechanics_chamber_id: str = eqx.field(static=True)
    volume_rate: VolumeRateLaw

    def __init__(
        self,
        name: str,
        mechanics_chamber_id: str,
        volume_rate: VolumeRateLaw,
        /,
        *,
        pressure_scale: float = 10.0,
        flow_scale: float = 1.0,
    ) -> None:
        chamber_id = str(mechanics_chamber_id).strip()
        if not chamber_id:
            raise ValueError("mechanics_chamber_id must be non-empty.")
        if not callable(volume_rate):
            raise TypeError("volume_rate must be callable.")
        p_scale = _positive_scalar(pressure_scale, "pressure_scale")[1]
        q_scale = _positive_scalar(flow_scale, "flow_scale")[1]
        component = DAEComponent(
            name,
            _two_port_variables(p_scale, q_scale),
            (
                DAEEquationBlock(
                    "equal_pressure",
                    _PressureEqualityResidual(),
                    _incidence(("pressure_in", 0), ("pressure_out", 0)),
                ),
                DAEEquationBlock(
                    "mechanics_volume_rate",
                    _MechanicsVolumeRateResidual(volume_rate),
                    _incidence(("flow_in", 0), ("flow_out", 0)),
                ),
            ),
            _two_ports(),
        )
        PressureFlowComponent.__init__(
            self,
            component,
            component_kind="mechanics-chamber-coupling",
            parameters=(("mechanics_chamber_id", chamber_id),),
            storage_owner=StorageOwner.MECHANICS,
        )
        self.mechanics_chamber_id = chamber_id
        self.volume_rate = volume_rate


def rc_pressure_transient(
    time: ArrayLike,
    initial_pressure: ArrayLike,
    source_pressure: ArrayLike,
    resistance: ArrayLike,
    compliance: ArrayLike,
    /,
) -> Array:
    """Analytic pressure of a series-R/shunt-C step response."""

    time_ = jnp.asarray(time)
    initial = jnp.asarray(initial_pressure)
    source = jnp.asarray(source_pressure)
    resistance_, _ = _positive_scalar(resistance, "resistance")
    compliance_, _ = _positive_scalar(compliance, "compliance")
    if bool(jnp.any(time_ < 0.0)):
        raise ValueError("time must be nonnegative.")
    return source + (initial - source) * jnp.exp(-time_ / (resistance_ * compliance_))


# Concise mathematical aliases.
R = Resistance
C = Compliance
L = Inertance
RCR = WindkesselRCR


__all__ = [
    "C",
    "Compliance",
    "ElastanceWaveform",
    "FlowSource",
    "FlowWaveform",
    "Inertance",
    "L",
    "MechanicsChamberCoupling",
    "PeriodicElastance",
    "PressureFlowComponent",
    "PressureSource",
    "PressureWaveform",
    "R",
    "RCR",
    "Resistance",
    "StorageOwner",
    "TimeVaryingElastance",
    "VolumeRateLaw",
    "WindkesselRCR",
    "rc_pressure_transient",
]
