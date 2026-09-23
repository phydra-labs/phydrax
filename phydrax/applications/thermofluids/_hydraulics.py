#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from enum import IntFlag

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._admissibility import AdmissibilityHeader, AdmissibilityReason
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._interpolation import linear_interpolate
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...dynamics import (
    DAEComponent,
    DAEDerivativeIncidence,
    DAEEquationBlock,
    DAEPort,
    DAEVariableBlock,
)
from ._process import HydraulicPortSpec, ThermofluidComponent


def _residual_numeric_id(semantic_id: str, parameters, /) -> str:
    return canonical_fingerprint(
        {
            "kind": "thermofluid-residual-binding",
            "semantic": semantic_id,
            "parameters": parameters,
        }
    )


class HydraulicReason(IntFlag):
    CAVITATION_MARGIN_EXCEEDED = 1 << 8
    REYNOLDS_LIMIT_EXCEEDED = 1 << 9
    CALIBRATION_SUPPORT_EXCEEDED = 1 << 10


class HydraulicFluidProperties(StrictModule, NonTrainableState):
    density: float = eqx.field(static=True)
    dynamic_viscosity: float = eqx.field(static=True)
    vapor_pressure: float = eqx.field(static=True)
    temperature: float = eqx.field(static=True)
    provenance: str = eqx.field(static=True)
    fluid_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        density: float,
        dynamic_viscosity: float,
        vapor_pressure: float,
        temperature: float,
        provenance: str,
    ) -> None:
        values = tuple(
            float(value)
            for value in (density, dynamic_viscosity, vapor_pressure, temperature)
        )
        provenance_ = str(provenance)
        if (
            any(not np.isfinite(value) for value in values)
            or values[0] <= 0.0
            or values[1] <= 0.0
            or values[2] < 0.0
            or values[3] <= 0.0
            or not provenance_
        ):
            raise ValueError("Hydraulic fluid properties are invalid.")
        self.density, self.dynamic_viscosity, self.vapor_pressure, self.temperature = (
            values
        )
        self.provenance = provenance_
        self.fluid_id = canonical_fingerprint(
            {
                "kind": "hydraulic-fluid-properties",
                "density": values[0],
                "dynamic_viscosity": values[1],
                "vapor_pressure": values[2],
                "temperature": values[3],
                "provenance": provenance_,
            }
        )


class HydraulicLawEvaluation(StrictModule):
    pressure_drop: Array
    reynolds_number: Array
    minimum_pressure: Array
    header: AdmissibilityHeader
    plan_id: str = eqx.field(static=True)


class HydraulicChannelPlan(StrictModule, NonTrainableState):
    fluid: HydraulicFluidProperties
    resistance: float = eqx.field(static=True)
    area: float = eqx.field(static=True)
    hydraulic_diameter: float = eqx.field(static=True)
    maximum_reynolds_number: float = eqx.field(static=True)
    geometry_kind: str = eqx.field(static=True)
    truncation_error_bound: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        fluid: HydraulicFluidProperties,
        /,
        *,
        resistance: float,
        area: float,
        hydraulic_diameter: float,
        maximum_reynolds_number: float,
        geometry_kind: str,
        truncation_error_bound: float = 0.0,
    ) -> None:
        values = tuple(
            float(value)
            for value in (
                resistance,
                area,
                hydraulic_diameter,
                maximum_reynolds_number,
                truncation_error_bound,
            )
        )
        kind = str(geometry_kind)
        if (
            not isinstance(fluid, HydraulicFluidProperties)
            or any(not np.isfinite(value) for value in values)
            or values[0] <= 0.0
            or values[1] <= 0.0
            or values[2] <= 0.0
            or values[3] <= 0.0
            or values[4] < 0.0
            or not kind
        ):
            raise ValueError("Hydraulic channel plan is invalid.")
        self.fluid = fluid
        self.resistance = values[0]
        self.area = values[1]
        self.hydraulic_diameter = values[2]
        self.maximum_reynolds_number = values[3]
        self.geometry_kind = kind
        self.truncation_error_bound = values[4]
        self.plan_id = canonical_fingerprint(
            {
                "kind": "hydraulic-channel",
                "fluid": fluid.fluid_id,
                "resistance": values[0],
                "area": values[1],
                "hydraulic_diameter": values[2],
                "maximum_reynolds_number": values[3],
                "geometry_kind": kind,
                "truncation_error_bound": values[4],
            }
        )

    @classmethod
    def circular(
        cls,
        fluid: HydraulicFluidProperties,
        /,
        *,
        radius: float,
        length: float,
        maximum_reynolds_number: float = 1000.0,
    ) -> HydraulicChannelPlan:
        radius_ = float(radius)
        length_ = float(length)
        if (
            not np.isfinite(radius_)
            or radius_ <= 0.0
            or not np.isfinite(length_)
            or length_ <= 0.0
        ):
            raise ValueError("Circular channel radius and length must be positive.")
        resistance = 8.0 * fluid.dynamic_viscosity * length_ / (np.pi * radius_**4)
        return cls(
            fluid,
            resistance=resistance,
            area=np.pi * radius_**2,
            hydraulic_diameter=2.0 * radius_,
            maximum_reynolds_number=maximum_reynolds_number,
            geometry_kind="circular-poiseuille",
        )

    @classmethod
    def rectangular(
        cls,
        fluid: HydraulicFluidProperties,
        /,
        *,
        width: float,
        height: float,
        length: float,
        series_terms: int = 32,
        maximum_truncation_error: float = 1.0e-8,
        maximum_reynolds_number: float = 1000.0,
    ) -> HydraulicChannelPlan:
        long_side = max(float(width), float(height))
        short_side = min(float(width), float(height))
        length_ = float(length)
        terms = int(series_terms)
        tolerance = float(maximum_truncation_error)
        if (
            not np.isfinite(long_side)
            or not np.isfinite(short_side)
            or not np.isfinite(length_)
            or min(long_side, short_side, length_) <= 0.0
            or terms <= 0
            or not np.isfinite(tolerance)
            or tolerance <= 0.0
        ):
            raise ValueError("Rectangular channel geometry or series policy is invalid.")
        odd = np.arange(1, 2 * terms, 2, dtype=np.float64)
        series = np.sum(np.tanh(odd * np.pi * long_side / (2.0 * short_side)) / odd**5)
        coefficient = 192.0 * short_side / (np.pi**5 * long_side)
        correction = 1.0 - coefficient * series
        tail_bound = coefficient / (4.0 * odd[-1] ** 4)
        relative_bound = tail_bound / correction
        if correction <= 0.0 or relative_bound > tolerance:
            raise ValueError(
                "Rectangular resistance series did not meet its error bound."
            )
        resistance = (
            12.0
            * fluid.dynamic_viscosity
            * length_
            / (long_side * short_side**3 * correction)
        )
        area = long_side * short_side
        hydraulic_diameter = 4.0 * area / (2.0 * (long_side + short_side))
        return cls(
            fluid,
            resistance=resistance,
            area=area,
            hydraulic_diameter=hydraulic_diameter,
            maximum_reynolds_number=maximum_reynolds_number,
            geometry_kind=f"rectangular-poiseuille-{terms}-terms",
            truncation_error_bound=relative_bound,
        )

    def evaluate(
        self,
        volume_flow: ArrayLike,
        left_pressure: ArrayLike,
        right_pressure: ArrayLike,
        /,
    ) -> HydraulicLawEvaluation:
        flow = jnp.asarray(volume_flow)
        left = jnp.asarray(left_pressure, dtype=flow.dtype)
        right = jnp.asarray(right_pressure, dtype=flow.dtype)
        speed = jnp.abs(flow) / self.area
        reynolds = (
            self.fluid.density
            * speed
            * self.hydraulic_diameter
            / self.fluid.dynamic_viscosity
        )
        minimum_pressure = jnp.minimum(left, right)
        finite = (
            jnp.isfinite(flow)
            & jnp.isfinite(left)
            & jnp.isfinite(right)
            & jnp.isfinite(reynolds)
        )
        pressure_ok = minimum_pressure > self.fluid.vapor_pressure
        reynolds_ok = reynolds <= self.maximum_reynolds_number
        supported = finite & pressure_ok & reynolds_ok
        reasons = jnp.asarray(0, dtype=jnp.uint32)
        reasons = jnp.where(
            finite,
            reasons,
            reasons | jnp.asarray(int(AdmissibilityReason.NONFINITE), jnp.uint32),
        )
        reasons = jnp.where(
            pressure_ok,
            reasons,
            reasons
            | jnp.asarray(int(HydraulicReason.CAVITATION_MARGIN_EXCEEDED), jnp.uint32),
        )
        reasons = jnp.where(
            reynolds_ok,
            reasons,
            reasons
            | jnp.asarray(int(HydraulicReason.REYNOLDS_LIMIT_EXCEEDED), jnp.uint32),
        )
        pressure_margin = (minimum_pressure - self.fluid.vapor_pressure) / jnp.maximum(
            jnp.maximum(jnp.abs(minimum_pressure), self.fluid.vapor_pressure),
            1.0,
        )
        reynolds_margin = (
            self.maximum_reynolds_number - reynolds
        ) / self.maximum_reynolds_number
        margin = jnp.minimum(pressure_margin, reynolds_margin)
        header = AdmissibilityHeader(
            jnp.where(supported, margin, jnp.minimum(margin, -1.0)),
            reasons,
            self.plan_id,
            canonical_fingerprint(
                {"kind": "hydraulic-channel-evidence", "plan": self.plan_id}
            ),
        )
        return HydraulicLawEvaluation(
            jnp.where(supported, self.resistance * flow, jnp.nan),
            reynolds,
            minimum_pressure,
            header,
            self.plan_id,
        )


class MonotoneHydraulicResponsePlan(StrictModule, NonTrainableState):
    fluid: HydraulicFluidProperties
    volume_flow: Array
    pressure_drop: Array
    reverse_symmetric: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        fluid: HydraulicFluidProperties,
        volume_flow: ArrayLike,
        pressure_drop: ArrayLike,
        /,
        *,
        reverse_symmetric: bool = False,
    ) -> None:
        flow = np.asarray(volume_flow, dtype=np.float64)
        pressure = np.asarray(pressure_drop, dtype=np.float64)
        symmetric = bool(reverse_symmetric)
        if (
            not isinstance(fluid, HydraulicFluidProperties)
            or flow.ndim != 1
            or flow.size < 2
            or pressure.shape != flow.shape
            or np.any(~np.isfinite(flow))
            or np.any(~np.isfinite(pressure))
            or np.any(np.diff(flow) <= 0.0)
            or np.any(np.diff(pressure) <= 0.0)
            or (symmetric and (flow[0] < 0.0 or pressure[0] < 0.0))
        ):
            raise ValueError("Calibrated hydraulic response must be finite and monotone.")
        self.fluid = fluid
        self.volume_flow = jnp.asarray(flow)
        self.pressure_drop = jnp.asarray(pressure)
        self.reverse_symmetric = symmetric
        self.plan_id = canonical_fingerprint(
            {
                "kind": "monotone-hydraulic-response",
                "fluid": fluid.fluid_id,
                "volume_flow": array_tree_fingerprint(flow),
                "pressure_drop": array_tree_fingerprint(pressure),
                "reverse_symmetric": symmetric,
            }
        )

    def evaluate(
        self,
        volume_flow: ArrayLike,
        left_pressure: ArrayLike,
        right_pressure: ArrayLike,
        /,
    ) -> HydraulicLawEvaluation:
        flow = jnp.asarray(volume_flow)
        left = jnp.asarray(left_pressure, dtype=flow.dtype)
        right = jnp.asarray(right_pressure, dtype=flow.dtype)
        coordinate = jnp.abs(flow) if self.reverse_symmetric else flow
        supported_flow = (coordinate >= self.volume_flow[0]) & (
            coordinate <= self.volume_flow[-1]
        )
        interpolation = linear_interpolate(
            self.volume_flow,
            self.pressure_drop,
            coordinate,
            bounds="fill",
            fill_value=jnp.nan,
        )
        interpolated = interpolation.values
        drop = jnp.sign(flow) * interpolated if self.reverse_symmetric else interpolated
        finite = jnp.isfinite(flow) & jnp.isfinite(left) & jnp.isfinite(right)
        pressure_ok = jnp.minimum(left, right) > self.fluid.vapor_pressure
        supported = finite & supported_flow & pressure_ok
        reasons = jnp.asarray(0, dtype=jnp.uint32)
        reasons = jnp.where(
            finite,
            reasons,
            reasons | jnp.asarray(int(AdmissibilityReason.NONFINITE), jnp.uint32),
        )
        reasons = jnp.where(
            supported_flow,
            reasons,
            reasons
            | jnp.asarray(int(HydraulicReason.CALIBRATION_SUPPORT_EXCEEDED), jnp.uint32),
        )
        reasons = jnp.where(
            pressure_ok,
            reasons,
            reasons
            | jnp.asarray(int(HydraulicReason.CAVITATION_MARGIN_EXCEEDED), jnp.uint32),
        )
        flow_scale = self.volume_flow[-1] - self.volume_flow[0]
        flow_margin = (
            jnp.minimum(
                coordinate - self.volume_flow[0], self.volume_flow[-1] - coordinate
            )
            / flow_scale
        )
        pressure_margin = (
            jnp.minimum(left, right) - self.fluid.vapor_pressure
        ) / jnp.maximum(
            jnp.maximum(jnp.abs(jnp.minimum(left, right)), self.fluid.vapor_pressure),
            1.0,
        )
        support_margin = jnp.minimum(flow_margin, pressure_margin)
        header = AdmissibilityHeader(
            jnp.where(supported, support_margin, jnp.minimum(support_margin, -1.0)),
            reasons,
            self.plan_id,
            canonical_fingerprint(
                {"kind": "hydraulic-response-evidence", "plan": self.plan_id}
            ),
        )
        return HydraulicLawEvaluation(
            jnp.where(supported, drop, jnp.nan),
            jnp.asarray(jnp.nan, dtype=flow.dtype),
            jnp.minimum(left, right),
            header,
            self.plan_id,
        )


class HydraulicReducedResponsePlan(StrictModule, NonTrainableState):
    """Fixed JAX scalar response with immutable support and artifact identity."""

    fluid: HydraulicFluidProperties
    response: Callable[[Array], Array] = eqx.field(static=True)
    minimum_flow: float = eqx.field(static=True)
    maximum_flow: float = eqx.field(static=True)
    reverse_symmetric: bool = eqx.field(static=True)
    artifact_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        fluid: HydraulicFluidProperties,
        response: Callable[[Array], Array],
        /,
        *,
        minimum_flow: float,
        maximum_flow: float,
        artifact_id: str,
        reverse_symmetric: bool = False,
    ) -> None:
        lower = float(minimum_flow)
        upper = float(maximum_flow)
        artifact = str(artifact_id)
        symmetric = bool(reverse_symmetric)
        if (
            not isinstance(fluid, HydraulicFluidProperties)
            or not callable(response)
            or not np.isfinite(lower)
            or not np.isfinite(upper)
            or lower >= upper
            or (symmetric and lower < 0.0)
            or not artifact
        ):
            raise ValueError("Hydraulic reduced response support is invalid.")
        self.fluid = fluid
        self.response = response
        self.minimum_flow = lower
        self.maximum_flow = upper
        self.reverse_symmetric = symmetric
        self.artifact_id = artifact
        self.plan_id = canonical_fingerprint(
            {
                "kind": "hydraulic-reduced-response",
                "fluid": fluid.fluid_id,
                "support": (lower, upper),
                "reverse_symmetric": symmetric,
                "artifact": artifact,
            }
        )

    def evaluate(
        self,
        volume_flow: ArrayLike,
        left_pressure: ArrayLike,
        right_pressure: ArrayLike,
        /,
    ) -> HydraulicLawEvaluation:
        flow = jnp.asarray(volume_flow)
        left = jnp.asarray(left_pressure, dtype=flow.dtype)
        right = jnp.asarray(right_pressure, dtype=flow.dtype)
        coordinate = jnp.abs(flow) if self.reverse_symmetric else flow
        supported_flow = (coordinate >= self.minimum_flow) & (
            coordinate <= self.maximum_flow
        )
        raw = jnp.asarray(self.response(coordinate), dtype=flow.dtype)
        if raw.shape != ():
            raise ValueError("Hydraulic reduced response must return one scalar.")
        drop = jnp.sign(flow) * raw if self.reverse_symmetric else raw
        pressure_ok = jnp.minimum(left, right) > self.fluid.vapor_pressure
        finite = (
            jnp.isfinite(flow)
            & jnp.isfinite(left)
            & jnp.isfinite(right)
            & jnp.isfinite(drop)
        )
        supported = finite & supported_flow & pressure_ok
        reasons = jnp.asarray(0, dtype=jnp.uint32)
        reasons = jnp.where(
            finite,
            reasons,
            reasons | jnp.asarray(int(AdmissibilityReason.NONFINITE), jnp.uint32),
        )
        reasons = jnp.where(
            supported_flow,
            reasons,
            reasons
            | jnp.asarray(int(HydraulicReason.CALIBRATION_SUPPORT_EXCEEDED), jnp.uint32),
        )
        reasons = jnp.where(
            pressure_ok,
            reasons,
            reasons
            | jnp.asarray(int(HydraulicReason.CAVITATION_MARGIN_EXCEEDED), jnp.uint32),
        )
        flow_scale = self.maximum_flow - self.minimum_flow
        flow_margin = (
            jnp.minimum(coordinate - self.minimum_flow, self.maximum_flow - coordinate)
            / flow_scale
        )
        pressure_margin = (
            jnp.minimum(left, right) - self.fluid.vapor_pressure
        ) / jnp.maximum(
            jnp.maximum(jnp.abs(jnp.minimum(left, right)), self.fluid.vapor_pressure),
            1.0,
        )
        support_margin = jnp.minimum(flow_margin, pressure_margin)
        header = AdmissibilityHeader(
            jnp.where(supported, support_margin, jnp.minimum(support_margin, -1.0)),
            reasons,
            self.plan_id,
            canonical_fingerprint(
                {"kind": "hydraulic-reduced-evidence", "plan": self.plan_id}
            ),
        )
        return HydraulicLawEvaluation(
            jnp.where(supported, drop, jnp.nan),
            jnp.asarray(jnp.nan, dtype=flow.dtype),
            jnp.minimum(left, right),
            header,
            self.plan_id,
        )


HydraulicPressureDropPlan = (
    HydraulicChannelPlan | HydraulicReducedResponsePlan | MonotoneHydraulicResponsePlan
)


def _hydraulic_port(name: str, fluid: HydraulicFluidProperties, /) -> HydraulicPortSpec:
    return HydraulicPortSpec(name, fluid_id=fluid.fluid_id)


def hydraulic_pressure_boundary_component(
    name: str,
    fluid: HydraulicFluidProperties,
    /,
    *,
    pressure: float,
) -> ThermofluidComponent:
    target = float(pressure)
    if not np.isfinite(target) or target <= fluid.vapor_pressure:
        raise ValueError("Hydraulic boundary pressure must exceed vapor pressure.")
    variables = (
        DAEVariableBlock("pressure", (), 0, state_scale=max(target, 1.0)),
        DAEVariableBlock("volume_flow", (), 0, state_scale=1.0, rate_scale=1.0),
    )

    def residual(time, jet, args):
        del time, args
        return jet.value("pressure") - target

    equation = DAEEquationBlock(
        "prescribed_pressure",
        residual,
        (DAEDerivativeIncidence("pressure"),),
        residual_semantic_id="thermofluids.hydraulics.pressure-boundary",
        residual_numeric_id=_residual_numeric_id(
            "thermofluids.hydraulics.pressure-boundary",
            {"pressure": target},
        ),
    )
    port = DAEPort("hydraulic", ("pressure",), ("volume_flow",))
    return ThermofluidComponent(
        DAEComponent(str(name), variables, (equation,), (port,)),
        (_hydraulic_port("hydraulic", fluid),),
        model_parameters=(("pressure", target), ("fluid_id", fluid.fluid_id)),
    )


def hydraulic_flow_boundary_component(
    name: str,
    fluid: HydraulicFluidProperties,
    /,
    *,
    volume_flow: float,
) -> ThermofluidComponent:
    target = float(volume_flow)
    if not np.isfinite(target):
        raise ValueError("Hydraulic boundary volume flow must be finite.")
    variables = (
        DAEVariableBlock("pressure", (), 0, state_scale=1.0e5, rate_scale=1.0e5),
        DAEVariableBlock("volume_flow", (), 0, state_scale=max(abs(target), 1.0)),
    )

    def residual(time, jet, args):
        del time, args
        return jet.value("volume_flow") - target

    equation = DAEEquationBlock(
        "prescribed_volume_flow",
        residual,
        (DAEDerivativeIncidence("volume_flow"),),
        residual_semantic_id="thermofluids.hydraulics.flow-boundary",
        residual_numeric_id=_residual_numeric_id(
            "thermofluids.hydraulics.flow-boundary",
            {"volume_flow": target},
        ),
    )
    port = DAEPort("hydraulic", ("pressure",), ("volume_flow",))
    return ThermofluidComponent(
        DAEComponent(str(name), variables, (equation,), (port,)),
        (_hydraulic_port("hydraulic", fluid),),
        model_parameters=(("volume_flow", target), ("fluid_id", fluid.fluid_id)),
    )


def hydraulic_channel_component(
    name: str,
    law: HydraulicPressureDropPlan,
    /,
) -> ThermofluidComponent:
    if not isinstance(
        law,
        (
            HydraulicChannelPlan,
            HydraulicReducedResponsePlan,
            MonotoneHydraulicResponsePlan,
        ),
    ):
        raise TypeError("Hydraulic channel requires a supported pressure-drop law.")
    variables = tuple(
        DAEVariableBlock(
            value,
            (),
            0,
            state_scale=1.0e5 if "pressure" in value else 1.0,
            rate_scale=1.0e5 if "pressure" in value else 1.0,
        )
        for value in (
            "left_pressure",
            "left_volume_flow",
            "right_pressure",
            "right_volume_flow",
        )
    )

    def pressure_drop_residual(time, jet, args):
        del time, args
        left = jet.value("left_pressure")
        right = jet.value("right_pressure")
        flow = jet.value("left_volume_flow")
        evaluation = law.evaluate(flow, left, right)
        return left - right - evaluation.pressure_drop

    def conservation_residual(time, jet, args):
        del time, args
        return jet.value("left_volume_flow") + jet.value("right_volume_flow")

    equations = (
        DAEEquationBlock(
            "pressure_drop",
            pressure_drop_residual,
            (
                DAEDerivativeIncidence("left_pressure"),
                DAEDerivativeIncidence("right_pressure"),
                DAEDerivativeIncidence("left_volume_flow"),
            ),
            residual_semantic_id="thermofluids.hydraulics.channel.pressure-drop",
            residual_numeric_id=_residual_numeric_id(
                "thermofluids.hydraulics.channel.pressure-drop",
                {"law_id": law.plan_id},
            ),
        ),
        DAEEquationBlock(
            "volume_conservation",
            conservation_residual,
            (
                DAEDerivativeIncidence("left_volume_flow"),
                DAEDerivativeIncidence("right_volume_flow"),
            ),
            residual_semantic_id="thermofluids.hydraulics.two-port.volume-conservation",
            residual_numeric_id=_residual_numeric_id(
                "thermofluids.hydraulics.two-port.volume-conservation",
                {},
            ),
        ),
    )
    ports = (
        DAEPort("left", ("left_pressure",), ("left_volume_flow",)),
        DAEPort("right", ("right_pressure",), ("right_volume_flow",)),
    )
    typed = (_hydraulic_port("left", law.fluid), _hydraulic_port("right", law.fluid))
    return ThermofluidComponent(
        DAEComponent(str(name), variables, equations, ports),
        typed,
        model_parameters=(("law_id", law.plan_id), ("fluid_id", law.fluid.fluid_id)),
    )


def hydraulic_compliance_component(
    name: str,
    fluid: HydraulicFluidProperties,
    /,
    *,
    compliance: float,
    reference_pressure: float,
    reference_volume: float = 0.0,
) -> ThermofluidComponent:
    compliance_ = float(compliance)
    pressure_ = float(reference_pressure)
    volume_ = float(reference_volume)
    if (
        not np.isfinite(compliance_)
        or compliance_ <= 0.0
        or not np.isfinite(pressure_)
        or pressure_ <= fluid.vapor_pressure
        or not np.isfinite(volume_)
        or volume_ < 0.0
    ):
        raise ValueError("Hydraulic compliance data are invalid.")
    variables = (
        DAEVariableBlock(
            "pressure", (), 1, state_scale=max(pressure_, 1.0), rate_scale=1.0
        ),
        DAEVariableBlock("volume_flow", (), 0, state_scale=1.0, rate_scale=1.0),
        DAEVariableBlock("volume", (), 0, state_scale=max(volume_, 1.0)),
    )

    def storage_residual(time, jet, args):
        del time, args
        return compliance_ * jet.value("pressure", 1) - jet.value("volume_flow")

    def constitutive_residual(time, jet, args):
        del time, args
        return (
            jet.value("volume")
            - volume_
            - compliance_ * (jet.value("pressure") - pressure_)
        )

    equations = (
        DAEEquationBlock(
            "compliance_storage",
            storage_residual,
            (
                DAEDerivativeIncidence("pressure", 1),
                DAEDerivativeIncidence("volume_flow"),
            ),
            residual_semantic_id="thermofluids.hydraulics.compliance.storage",
            residual_numeric_id=_residual_numeric_id(
                "thermofluids.hydraulics.compliance.storage",
                {"compliance": compliance_},
            ),
        ),
        DAEEquationBlock(
            "compliance_constitutive",
            constitutive_residual,
            (
                DAEDerivativeIncidence("volume"),
                DAEDerivativeIncidence("pressure"),
            ),
            residual_semantic_id="thermofluids.hydraulics.compliance.constitutive",
            residual_numeric_id=_residual_numeric_id(
                "thermofluids.hydraulics.compliance.constitutive",
                {
                    "compliance": compliance_,
                    "reference_pressure": pressure_,
                    "reference_volume": volume_,
                },
            ),
        ),
    )
    port = DAEPort("hydraulic", ("pressure",), ("volume_flow",))
    return ThermofluidComponent(
        DAEComponent(str(name), variables, equations, (port,)),
        (_hydraulic_port("hydraulic", fluid),),
        model_parameters=(
            ("compliance", compliance_),
            ("reference_pressure", pressure_),
            ("reference_volume", volume_),
        ),
    )


def hydraulic_inertance_component(
    name: str,
    fluid: HydraulicFluidProperties,
    /,
    *,
    inertance: float,
) -> ThermofluidComponent:
    inertance_ = float(inertance)
    if not np.isfinite(inertance_) or inertance_ <= 0.0:
        raise ValueError("Hydraulic inertance must be finite and positive.")
    variables = (
        DAEVariableBlock("left_pressure", (), 0, state_scale=1.0e5, rate_scale=1.0e5),
        DAEVariableBlock("left_volume_flow", (), 1, state_scale=1.0, rate_scale=1.0),
        DAEVariableBlock("right_pressure", (), 0, state_scale=1.0e5, rate_scale=1.0e5),
        DAEVariableBlock("right_volume_flow", (), 0, state_scale=1.0, rate_scale=1.0),
    )

    def momentum_residual(time, jet, args):
        del time, args
        return (
            jet.value("left_pressure")
            - jet.value("right_pressure")
            - inertance_ * jet.value("left_volume_flow", 1)
        )

    def conservation_residual(time, jet, args):
        del time, args
        return jet.value("left_volume_flow") + jet.value("right_volume_flow")

    equations = (
        DAEEquationBlock(
            "inertance_momentum",
            momentum_residual,
            (
                DAEDerivativeIncidence("left_pressure"),
                DAEDerivativeIncidence("right_pressure"),
                DAEDerivativeIncidence("left_volume_flow", 1),
            ),
            residual_semantic_id="thermofluids.hydraulics.inertance.momentum",
            residual_numeric_id=_residual_numeric_id(
                "thermofluids.hydraulics.inertance.momentum",
                {"inertance": inertance_},
            ),
        ),
        DAEEquationBlock(
            "volume_conservation",
            conservation_residual,
            (
                DAEDerivativeIncidence("left_volume_flow"),
                DAEDerivativeIncidence("right_volume_flow"),
            ),
            residual_semantic_id="thermofluids.hydraulics.two-port.volume-conservation",
            residual_numeric_id=_residual_numeric_id(
                "thermofluids.hydraulics.two-port.volume-conservation",
                {},
            ),
        ),
    )
    ports = (
        DAEPort("left", ("left_pressure",), ("left_volume_flow",)),
        DAEPort("right", ("right_pressure",), ("right_volume_flow",)),
    )
    return ThermofluidComponent(
        DAEComponent(str(name), variables, equations, ports),
        (_hydraulic_port("left", fluid), _hydraulic_port("right", fluid)),
        model_parameters=(("inertance", inertance_),),
    )


def hydraulic_junction_component(
    name: str,
    fluid: HydraulicFluidProperties,
    port_count: int,
    /,
) -> ThermofluidComponent:
    count = int(port_count)
    if count < 2:
        raise ValueError("Hydraulic junction requires at least two ports.")
    variables = []
    ports = []
    typed = []
    for index in range(count):
        prefix = f"port_{index}"
        variables.extend(
            (
                DAEVariableBlock(f"{prefix}_pressure", (), 0, state_scale=1.0e5),
                DAEVariableBlock(f"{prefix}_volume_flow", (), 0, state_scale=1.0),
            )
        )
        ports.append(
            DAEPort(
                prefix,
                (f"{prefix}_pressure",),
                (f"{prefix}_volume_flow",),
            )
        )
        typed.append(_hydraulic_port(prefix, fluid))

    def pressure_residual(index):
        def residual(time, jet, args):
            del time, args
            return jet.value(f"port_{index}_pressure") - jet.value("port_0_pressure")

        return residual

    equations = [
        DAEEquationBlock(
            f"equal_pressure_{index}",
            pressure_residual(index),
            (
                DAEDerivativeIncidence(f"port_{index}_pressure"),
                DAEDerivativeIncidence("port_0_pressure"),
            ),
            residual_semantic_id="thermofluids.hydraulics.junction.equal-pressure",
            residual_numeric_id=_residual_numeric_id(
                "thermofluids.hydraulics.junction.equal-pressure",
                {"port_index": index},
            ),
        )
        for index in range(1, count)
    ]

    def flow_residual(time, jet, args):
        del time, args
        return sum(jet.value(f"port_{index}_volume_flow") for index in range(count))

    equations.append(
        DAEEquationBlock(
            "volume_conservation",
            flow_residual,
            tuple(
                DAEDerivativeIncidence(f"port_{index}_volume_flow")
                for index in range(count)
            ),
            residual_semantic_id="thermofluids.hydraulics.junction.volume-conservation",
            residual_numeric_id=_residual_numeric_id(
                "thermofluids.hydraulics.junction.volume-conservation",
                {"port_count": count},
            ),
        )
    )
    return ThermofluidComponent(
        DAEComponent(str(name), tuple(variables), tuple(equations), tuple(ports)),
        tuple(typed),
        model_parameters=(("port_count", count), ("fluid_id", fluid.fluid_id)),
    )


__all__ = [
    "HydraulicChannelPlan",
    "HydraulicFluidProperties",
    "HydraulicLawEvaluation",
    "HydraulicPressureDropPlan",
    "HydraulicReason",
    "HydraulicReducedResponsePlan",
    "MonotoneHydraulicResponsePlan",
    "hydraulic_channel_component",
    "hydraulic_compliance_component",
    "hydraulic_flow_boundary_component",
    "hydraulic_inertance_component",
    "hydraulic_junction_component",
    "hydraulic_pressure_boundary_component",
]
