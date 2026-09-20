#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._diffusion import (
    ConservativeAdvectionPlan,
    ConservativeBoundaryCondition,
    ConservativeDiffusionPlan,
    PreparedConservativeAdvection,
    PreparedConservativeDiffusion,
)
from ._incompressible import FaceVelocity, PreparedMACOperators
from ._precision import FiniteVolumePrecisionPolicy


MACThermalBoundaryKind: TypeAlias = Literal[
    "periodic", "temperature", "adiabatic", "heat_flux"
]
MACEnthalpyAdvection: TypeAlias = Literal["centered", "upwind"]


def _finite(value: ArrayLike, owner: str, /) -> Array:
    array = jnp.asarray(value)
    return eqx.error_if(array, jnp.any(~jnp.isfinite(array)), f"{owner} must be finite.")


def _boundary_slice(value: Array, axis: int, index: int, /) -> Array:
    location = [slice(None)] * value.ndim
    location[axis] = index
    return value[tuple(location)]


class MACThermalBoundaryCondition(StrictModule, NonTrainableState):
    kind: MACThermalBoundaryKind = eqx.field(static=True)
    value: Array
    function: Any = eqx.field(static=True)
    function_id: str = eqx.field(static=True)
    boundary_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: MACThermalBoundaryKind,
        value: ArrayLike | Any = 0.0,
        /,
        *,
        function_id: str | None = None,
    ):
        if kind not in ("periodic", "temperature", "adiabatic", "heat_flux"):
            raise ValueError("Unknown MAC thermal boundary kind.")
        if callable(value):
            if kind not in ("temperature", "heat_flux"):
                raise ValueError(
                    "Dynamic MAC thermal data require temperature or heat_flux."
                )
            identifier = "" if function_id is None else str(function_id)
            if not identifier:
                raise ValueError("Dynamic MAC thermal data require a function_id.")
            value_ = jnp.asarray(0.0)
            function = value
            host_value: Any = "dynamic"
        else:
            if function_id is not None:
                raise ValueError("Static MAC thermal data cannot carry function_id.")
            value_ = jnp.asarray(value)
            host = np.asarray(value_)
            if np.iscomplexobj(host) or np.any(~np.isfinite(host)):
                raise ValueError("MAC thermal boundary data must be finite and real.")
            function = None
            identifier = "none"
            host_value = host.tolist()
        if kind in ("periodic", "adiabatic") and (
            function is not None or value_.shape != () or float(value_) != 0.0
        ):
            raise ValueError(f"{kind} MAC thermal boundaries cannot carry data.")
        self.kind = kind
        self.value = value_
        self.function = function
        self.function_id = identifier
        self.boundary_id = canonical_fingerprint(
            {
                "kind": "mac-thermal-boundary",
                "boundary_kind": kind,
                "value": host_value,
                "function": identifier,
                "heat_flux_sign": "outward-loss",
            }
        )

    def evaluate(self, time: Array, coordinates: Array, args: Any = None, /) -> Array:
        target_shape = coordinates.shape[:-1]
        if self.function is None:
            output = jnp.broadcast_to(
                jnp.asarray(self.value, dtype=coordinates.dtype), target_shape
            )
        else:
            output = jnp.asarray(
                self.function(time, coordinates, args), dtype=coordinates.dtype
            )
            if output.shape == ():
                output = jnp.broadcast_to(output, target_shape)
            elif output.shape != target_shape:
                raise ValueError(
                    f"Dynamic MAC thermal data must be scalar or match boundary shape {target_shape}."
                )
        return _finite(output, "MAC thermal boundary evaluation")


class MACThermalBoundarySet(StrictModule, NonTrainableState):
    operators: PreparedMACOperators
    conditions: tuple[
        tuple[MACThermalBoundaryCondition, MACThermalBoundaryCondition], ...
    ]
    boundary_id: str = eqx.field(static=True)

    def __init__(
        self,
        operators: PreparedMACOperators,
        /,
        *,
        walls: Mapping[
            str,
            tuple[
                MACThermalBoundaryCondition | MACThermalBoundaryKind,
                MACThermalBoundaryCondition | MACThermalBoundaryKind,
            ],
        ]
        | None = None,
    ):
        if not isinstance(operators, PreparedMACOperators):
            raise TypeError("operators must be PreparedMACOperators.")
        supplied = (
            {} if walls is None else {str(name): value for name, value in walls.items()}
        )
        grid = operators.discretization.grid
        unknown = set(supplied).difference(grid.axis_names)
        if unknown:
            raise ValueError(
                f"MAC thermal walls reference unknown axes {sorted(unknown)!r}."
            )
        conditions = []
        for axis_name, axis in zip(grid.axis_names, grid.structured_axes, strict=True):
            if axis.periodic:
                if axis_name in supplied:
                    raise ValueError("Periodic MAC axes do not accept thermal wall data.")
                pair = (
                    MACThermalBoundaryCondition("periodic"),
                    MACThermalBoundaryCondition("periodic"),
                )
            else:
                raw = supplied.get(axis_name, ("adiabatic", "adiabatic"))
                if len(raw) != 2:
                    raise ValueError(
                        "Each MAC thermal wall axis requires lower/upper data."
                    )
                pair = tuple(
                    value
                    if isinstance(value, MACThermalBoundaryCondition)
                    else MACThermalBoundaryCondition(value)
                    for value in raw
                )
                if any(value.kind == "periodic" for value in pair):
                    raise ValueError("Static MAC walls cannot use periodic thermal data.")
                axis_index = grid.axis_names.index(axis_name)
                expected = grid.shape[:axis_index] + grid.shape[axis_index + 1 :]
                for condition in pair:
                    if condition.value.shape not in ((), expected):
                        raise ValueError(
                            f"MAC thermal wall data must be scalar or match tangential shape {expected}."
                        )
            conditions.append(pair)
        self.operators = operators
        self.conditions = tuple(conditions)
        self.boundary_id = canonical_fingerprint(
            {
                "kind": "mac-thermal-boundary-set",
                "operators": operators.prepared_id,
                "conditions": [
                    (lower.boundary_id, upper.boundary_id)
                    for lower, upper in self.conditions
                ],
            }
        )

    def diffusion_conditions(
        self,
    ) -> dict[str, tuple[ConservativeBoundaryCondition, ConservativeBoundaryCondition]]:
        return {
            axis_name: tuple(
                ConservativeBoundaryCondition(
                    "periodic"
                    if condition.kind == "periodic"
                    else ("dirichlet" if condition.kind == "temperature" else "neumann")
                )
                for condition in pair
            )
            for axis_name, pair in zip(
                self.operators.discretization.grid.axis_names,
                self.conditions,
                strict=True,
            )
        }

    def temperature_values(
        self, time: Array, args: Any = None, /
    ) -> dict[str, tuple[Array, Array]]:
        discretization = self.operators.discretization
        output = {}
        for axis, (axis_name, pair) in enumerate(
            zip(discretization.grid.axis_names, self.conditions, strict=True)
        ):
            values = []
            for index, condition in ((0, pair[0]), (-1, pair[1])):
                coordinates = jnp.take(
                    discretization.face_centers[axis], index, axis=axis
                )
                values.append(
                    condition.evaluate(time, coordinates, args)
                    if condition.kind == "temperature"
                    else jnp.asarray(0.0, dtype=coordinates.dtype)
                )
            output[axis_name] = tuple(values)
        return output


class MACEnthalpyFluxResult(StrictModule):
    enthalpy_face_values: tuple[Array, ...]
    advective_fluxes: tuple[Array, ...]
    conductive_fluxes: tuple[Array, ...]
    boundary_heat_fluxes: tuple[Array, ...]
    advective_divergence: Array
    conductive_divergence: Array
    boundary_heat_divergence: Array
    source: Array
    rate: Array
    finite: Array
    successful: Array
    transport_id: str = eqx.field(static=True)
    grid_id: str = eqx.field(static=True)


class MACEnthalpyDiagnostics(StrictModule):
    total_enthalpy: Array
    enthalpy_rate: Array
    advective_content_rate: Array
    conductive_content_rate: Array
    boundary_heat_rate: Array
    source_power: Array
    balance_defect: Array
    minimum_temperature: Array
    maximum_temperature: Array
    minimum_liquid_fraction: Array
    maximum_liquid_fraction: Array
    finite: Array
    successful: Array
    transport_id: str = eqx.field(static=True)
    grid_id: str = eqx.field(static=True)


class MACEnthalpyStepRestriction(StrictModule):
    advective: Array
    diffusive: Array
    selected: Array
    finite: Array
    successful: Array
    transport_id: str = eqx.field(static=True)
    grid_id: str = eqx.field(static=True)


class MACEnthalpyTransportPlan(StrictModule, NonTrainableState):
    operators: PreparedMACOperators
    boundaries: MACThermalBoundarySet
    advection: MACEnthalpyAdvection = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        operators: PreparedMACOperators,
        boundaries: MACThermalBoundarySet,
        /,
        *,
        advection: MACEnthalpyAdvection = "upwind",
    ):
        if not isinstance(operators, PreparedMACOperators):
            raise TypeError("operators must be PreparedMACOperators.")
        if not isinstance(boundaries, MACThermalBoundarySet):
            raise TypeError("boundaries must be MACThermalBoundarySet.")
        if boundaries.operators.prepared_id != operators.prepared_id:
            raise ValueError("MAC enthalpy boundaries must share prepared operators.")
        if advection not in ("centered", "upwind"):
            raise ValueError("MAC enthalpy advection must be centered or upwind.")
        self.operators = operators
        self.boundaries = boundaries
        self.advection = advection
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mac-enthalpy-transport-plan",
                "operators": operators.prepared_id,
                "boundaries": boundaries.boundary_id,
                "advection": advection,
            }
        )

    def prepare(self) -> PreparedMACEnthalpyTransport:
        return PreparedMACEnthalpyTransport(self)


class PreparedMACEnthalpyTransport(StrictModule, NonTrainableState):
    plan: MACEnthalpyTransportPlan
    advection: PreparedConservativeAdvection
    diffusion: PreparedConservativeDiffusion
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: MACEnthalpyTransportPlan, /):
        if not isinstance(plan, MACEnthalpyTransportPlan):
            raise TypeError("plan must be MACEnthalpyTransportPlan.")
        operators = plan.operators
        discretization = operators.discretization
        zero_velocity = tuple(
            jnp.zeros(layout.shape, dtype=operators.pressure_space.dtype)
            for layout in discretization.face_layouts
        )
        precision = FiniteVolumePrecisionPolicy(
            np.dtype(operators.pressure_space.dtype).name
        )
        conditions = plan.boundaries.diffusion_conditions()
        advection = ConservativeAdvectionPlan(
            discretization.grid,
            form="conservative",
            reconstruction="arithmetic" if plan.advection == "centered" else "upwind",
            boundaries=conditions,
            precision=precision,
        ).prepare(zero_velocity)
        diffusion = ConservativeDiffusionPlan(
            discretization.grid,
            boundaries=conditions,
            interpolation="harmonic",
            precision=precision,
        ).prepare(
            jnp.ones(discretization.cell_shape, dtype=operators.pressure_space.dtype)
        )
        self.plan = plan
        self.advection = advection
        self.diffusion = diffusion
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-mac-enthalpy-transport",
                "plan": plan.plan_id,
                "advection": advection.prepared_id,
                "diffusion": diffusion.operator_id,
            }
        )

    @property
    def operators(self) -> PreparedMACOperators:
        return self.plan.operators

    def _validate_cell(self, value: ArrayLike, owner: str, /) -> Array:
        array = jnp.asarray(value)
        expected = self.operators.discretization.cell_shape
        if array.shape != expected:
            raise ValueError(
                f"{owner} must have cell shape {expected}; got {array.shape}."
            )
        if array.dtype != self.operators.pressure_space.dtype:
            raise TypeError(
                f"{owner} must have dtype {self.operators.pressure_space.dtype}; got {array.dtype}."
            )
        return _finite(array, owner)

    def _validate_velocity(self, velocity: FaceVelocity, /) -> FaceVelocity:
        values = self.operators.validate_velocity(velocity)
        for axis, grid_axis in enumerate(
            self.operators.discretization.grid.structured_axes
        ):
            if grid_axis.periodic:
                continue
            defect = jnp.maximum(
                jnp.max(jnp.abs(_boundary_slice(values[axis], axis, 0))),
                jnp.max(jnp.abs(_boundary_slice(values[axis], axis, -1))),
            )
            values = (
                values[:axis]
                + (
                    eqx.error_if(
                        values[axis],
                        defect > 0.0,
                        "MAC enthalpy transport requires impermeable walls.",
                    ),
                )
                + values[axis + 1 :]
            )
        return tuple(_finite(value, "MAC enthalpy velocity") for value in values)

    def _prescribed_heat_fluxes(
        self,
        time: Array,
        fluxes: tuple[Array, ...],
        args: Any,
        /,
    ) -> tuple[Array, ...]:
        output = list(fluxes)
        discretization = self.operators.discretization
        for axis, pair in enumerate(self.plan.boundaries.conditions):
            for index, condition, orientation in (
                (0, pair[0], 1.0),
                (-1, pair[1], -1.0),
            ):
                if condition.kind != "heat_flux":
                    continue
                coordinates = jnp.take(
                    discretization.face_centers[axis], index, axis=axis
                )
                outward_loss = condition.evaluate(time, coordinates, args)
                location = [slice(None)] * output[axis].ndim
                location[axis] = index
                output[axis] = (
                    output[axis].at[tuple(location)].set(orientation * outward_loss)
                )
        return tuple(output)

    def evaluate(
        self,
        time: ArrayLike,
        enthalpy: ArrayLike,
        temperature: ArrayLike,
        conductivity: ArrayLike,
        velocity: FaceVelocity,
        source: ArrayLike,
        args: Any = None,
        /,
    ) -> MACEnthalpyFluxResult:
        time_ = _finite(jnp.asarray(time), "MAC enthalpy stage time")
        if time_.shape != ():
            raise ValueError("MAC enthalpy stage time must be scalar.")
        enthalpy_ = self._validate_cell(enthalpy, "MAC volumetric enthalpy")
        temperature_ = self._validate_cell(temperature, "MAC temperature")
        conductivity_ = self._validate_cell(conductivity, "MAC conductivity")
        source_ = self._validate_cell(source, "MAC enthalpy source")
        conductivity_ = eqx.error_if(
            conductivity_,
            jnp.any(conductivity_ <= 0.0),
            "MAC conductivity must be positive.",
        )
        velocity_ = self._validate_velocity(velocity)
        boundary_values = self.plan.boundaries.temperature_values(time_, args)
        enthalpy_face_values = self.advection.face_values(
            enthalpy_,
            velocity=velocity_,
            boundary_values={
                name: (jnp.asarray(0.0), jnp.asarray(0.0))
                for name in self.operators.discretization.grid.axis_names
            },
        )
        advective_fluxes = tuple(
            face_velocity * face_value
            for face_velocity, face_value in zip(
                velocity_, enthalpy_face_values, strict=True
            )
        )
        advective_divergence = self.advection.divergence(advective_fluxes)
        internal_heat_fluxes = self.diffusion.fluxes(
            temperature_, conductivity_, boundary_values
        )
        conductive_fluxes = self._prescribed_heat_fluxes(
            time_, internal_heat_fluxes, args
        )
        boundary_heat_fluxes = tuple(
            total - internal
            for total, internal in zip(
                conductive_fluxes, internal_heat_fluxes, strict=True
            )
        )
        internal_divergence = self.diffusion.divergence(internal_heat_fluxes)
        boundary_divergence = self.diffusion.divergence(boundary_heat_fluxes)
        conductive_divergence = internal_divergence + boundary_divergence
        rate = _finite(
            -advective_divergence + conductive_divergence + source_,
            "MAC enthalpy rate",
        )
        finite = jnp.all(
            jnp.stack(
                (
                    jnp.all(jnp.isfinite(advective_divergence)),
                    jnp.all(jnp.isfinite(conductive_divergence)),
                    jnp.all(jnp.isfinite(rate)),
                    *(jnp.all(jnp.isfinite(value)) for value in advective_fluxes),
                    *(jnp.all(jnp.isfinite(value)) for value in conductive_fluxes),
                )
            )
        )
        return MACEnthalpyFluxResult(
            enthalpy_face_values,
            advective_fluxes,
            conductive_fluxes,
            boundary_heat_fluxes,
            advective_divergence,
            conductive_divergence,
            boundary_divergence,
            source_,
            rate,
            finite,
            finite,
            self.prepared_id,
            self.operators.discretization.grid.prepared_id,
        )

    def diagnostics(
        self,
        enthalpy: ArrayLike,
        temperature: ArrayLike,
        liquid_fraction: ArrayLike,
        result: MACEnthalpyFluxResult,
        /,
    ) -> MACEnthalpyDiagnostics:
        if (
            not isinstance(result, MACEnthalpyFluxResult)
            or result.transport_id != self.prepared_id
        ):
            raise ValueError("MAC enthalpy diagnostics provenance does not match.")
        enthalpy_ = self._validate_cell(enthalpy, "MAC volumetric enthalpy")
        temperature_ = self._validate_cell(temperature, "MAC temperature")
        fraction = self._validate_cell(liquid_fraction, "MAC liquid fraction")
        volumes = self.operators.discretization.cell_volumes
        total = jnp.sum(volumes * enthalpy_)
        rate = jnp.sum(volumes * result.rate)
        advective = -jnp.sum(volumes * result.advective_divergence)
        conductive = jnp.sum(
            volumes * (result.conductive_divergence - result.boundary_heat_divergence)
        )
        boundary = jnp.sum(volumes * result.boundary_heat_divergence)
        source = jnp.sum(volumes * result.source)
        defect = rate - (advective + conductive + boundary + source)
        values = jnp.stack(
            (
                total,
                rate,
                advective,
                conductive,
                boundary,
                source,
                defect,
                jnp.min(temperature_),
                jnp.max(temperature_),
                jnp.min(fraction),
                jnp.max(fraction),
            )
        )
        finite = result.finite & jnp.all(jnp.isfinite(values))
        return MACEnthalpyDiagnostics(
            *tuple(values),
            finite,
            finite,
            self.prepared_id,
            self.operators.discretization.grid.prepared_id,
        )

    def step_restriction(
        self,
        velocity: FaceVelocity,
        conductivity: ArrayLike,
        temperature_enthalpy_derivative: ArrayLike,
        /,
    ) -> MACEnthalpyStepRestriction:
        velocity_ = self._validate_velocity(velocity)
        conductivity_ = self._validate_cell(conductivity, "MAC conductivity")
        derivative = self._validate_cell(
            temperature_enthalpy_derivative,
            "MAC temperature-enthalpy derivative",
        )
        discretization = self.operators.discretization
        inverse_advective = jnp.zeros(
            discretization.cell_shape, dtype=self.operators.pressure_space.dtype
        )
        for axis_index, axis in enumerate(discretization.grid.structured_axes):
            oriented = velocity_[axis_index] * discretization.face_measures[axis_index]
            if axis.periodic:
                lower = oriented
                upper = jnp.roll(oriented, -1, axis=axis_index)
            else:
                lower_location = [slice(None)] * oriented.ndim
                upper_location = [slice(None)] * oriented.ndim
                lower_location[axis_index] = slice(0, oriented.shape[axis_index] - 1)
                upper_location[axis_index] = slice(1, oriented.shape[axis_index])
                lower = oriented[tuple(lower_location)]
                upper = oriented[tuple(upper_location)]
            inverse_advective = (
                inverse_advective
                + (jnp.maximum(-lower, 0.0) + jnp.maximum(upper, 0.0))
                / discretization.cell_volumes
            )
        advective_rate = jnp.max(inverse_advective)
        advective = jnp.where(advective_rate > 0.0, 1.0 / advective_rate, jnp.inf)
        effective_diffusivity = conductivity_ * derivative
        upper_diffusivity = jnp.full_like(
            effective_diffusivity, jnp.max(effective_diffusivity)
        )
        diagonal = self.diffusion.diagonal_with_coefficient(upper_diffusivity)
        diffusive_rate = jnp.max(jnp.maximum(0.0, -diagonal))
        diffusive = jnp.where(diffusive_rate > 0.0, 1.0 / diffusive_rate, jnp.inf)
        selected = jnp.minimum(advective, diffusive)
        finite = ~jnp.isnan(selected)
        return MACEnthalpyStepRestriction(
            advective,
            diffusive,
            selected,
            finite,
            finite,
            self.prepared_id,
            discretization.grid.prepared_id,
        )


__all__ = [
    "MACEnthalpyAdvection",
    "MACEnthalpyDiagnostics",
    "MACEnthalpyFluxResult",
    "MACEnthalpyStepRestriction",
    "MACEnthalpyTransportPlan",
    "MACThermalBoundaryCondition",
    "MACThermalBoundaryKind",
    "MACThermalBoundarySet",
    "PreparedMACEnthalpyTransport",
]
