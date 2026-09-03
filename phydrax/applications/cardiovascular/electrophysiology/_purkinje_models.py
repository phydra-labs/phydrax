#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Qualified reduced human Purkinje-fibre ionic dynamics.

This Stewart-2009-informed subsystem retains the Purkinje fast upstroke,
plateau calcium, transient/delayed potassium currents, ``I_f`` automaticity,
pumps/exchange, and a one-pool SR.  It has its own schema and coefficients;
it is not a ventricular model selected by a phenotype flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntFlag
from math import isfinite
from typing import ClassVar, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ._membrane_scaling import CardiacMembraneScaling
from ._reaction import (
    ArrayLike,
    CardiacReactionEvaluation,
    CardiacReactionParameterLayout,
    CardiacReactionStateLayout,
)


class PurkinjePhenotype(Enum):
    """Qualified Purkinje cellular identity."""

    HUMAN_STEWART2009_REDUCED = "human-purkinje-stewart2009-reduced"


class PurkinjeAdmissibilityStatus(IntFlag):
    """Fail-closed Purkinje state status bits."""

    SUCCESS = 0
    NONFINITE = 1
    GATE_OUT_OF_RANGE = 2
    NONPOSITIVE_CALCIUM = 4
    VOLTAGE_OUT_OF_RANGE = 8


def _positive(value: float, name: str, /) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real scalar, not bool.")
    resolved = float(value)
    if not isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return resolved


def _nonnegative(value: float, name: str, /) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real scalar, not bool.")
    resolved = float(value)
    if not isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return resolved


def _finite(value: float, name: str, /) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real scalar, not bool.")
    resolved = float(value)
    if not isfinite(resolved):
        raise ValueError(f"{name} must be finite.")
    return resolved


def _exp(value: Array, /) -> Array:
    return jnp.exp(jnp.clip(value, -80.0, 80.0))


def _shape(value: Sequence[int], /) -> tuple[int, ...]:
    shape = tuple(value)
    if any(
        isinstance(axis, bool) or not isinstance(axis, int) or axis < 0 for axis in shape
    ):
        raise ValueError("batch_shape axes must be nonnegative integers.")
    return shape


class StewartPurkinjeParameters(StrictModule, NonTrainableState):
    """Coefficient plan for the Stewart-informed human Purkinje subsystem."""

    phenotype: PurkinjePhenotype = eqx.field(static=True)
    rtf_mV: float = eqx.field(static=True)
    sodium_i_mM: float = eqx.field(static=True)
    sodium_o_mM: float = eqx.field(static=True)
    potassium_i_mM: float = eqx.field(static=True)
    potassium_o_mM: float = eqx.field(static=True)
    calcium_o_mM: float = eqx.field(static=True)
    funny_reversal_mV: float = eqx.field(static=True)
    g_na: float = eqx.field(static=True)
    g_cal: float = eqx.field(static=True)
    g_to: float = eqx.field(static=True)
    g_kr: float = eqx.field(static=True)
    g_ks: float = eqx.field(static=True)
    g_k1: float = eqx.field(static=True)
    g_f: float = eqx.field(static=True)
    g_bna: float = eqx.field(static=True)
    g_bca: float = eqx.field(static=True)
    i_nak_max: float = eqx.field(static=True)
    i_naca_max: float = eqx.field(static=True)
    ca_current_scale: float = eqx.field(static=True)
    uptake_max: float = eqx.field(static=True)
    uptake_half_mM: float = eqx.field(static=True)
    sr_leak_rate: float = eqx.field(static=True)
    sr_release_rate: float = eqx.field(static=True)
    sr_volume_ratio: float = eqx.field(static=True)
    cytosolic_buffer_factor: float = eqx.field(static=True)
    parameter_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        rtf_mV: float = 26.7137607,
        sodium_i_mM: float = 9.44,
        sodium_o_mM: float = 140.0,
        potassium_i_mM: float = 141.2,
        potassium_o_mM: float = 5.4,
        calcium_o_mM: float = 2.0,
        funny_reversal_mV: float = -30.0,
        g_na: float = 5.6718,
        g_cal: float = 0.060,
        g_to: float = 0.120,
        g_kr: float = 0.020,
        g_ks: float = 0.030,
        g_k1: float = 0.010,
        g_f: float = 0.012,
        g_bna: float = 0.00030,
        g_bca: float = 0.00010,
        i_nak_max: float = 0.30,
        i_naca_max: float = 1000.0,
        ca_current_scale: float = 2.5e-5,
        uptake_max: float = 0.0030,
        uptake_half_mM: float = 0.0007,
        sr_leak_rate: float = 2.0e-4,
        sr_release_rate: float = 0.20,
        sr_volume_ratio: float = 10.0,
        cytosolic_buffer_factor: float = 0.12,
    ):
        values = {
            "rtf_mV": _positive(rtf_mV, "rtf_mV"),
            "sodium_i_mM": _positive(sodium_i_mM, "sodium_i_mM"),
            "sodium_o_mM": _positive(sodium_o_mM, "sodium_o_mM"),
            "potassium_i_mM": _positive(potassium_i_mM, "potassium_i_mM"),
            "potassium_o_mM": _positive(potassium_o_mM, "potassium_o_mM"),
            "calcium_o_mM": _positive(calcium_o_mM, "calcium_o_mM"),
            "funny_reversal_mV": _finite(funny_reversal_mV, "funny_reversal_mV"),
            "g_na": _positive(g_na, "g_na"),
            "g_cal": _positive(g_cal, "g_cal"),
            "g_to": _positive(g_to, "g_to"),
            "g_kr": _positive(g_kr, "g_kr"),
            "g_ks": _positive(g_ks, "g_ks"),
            "g_k1": _positive(g_k1, "g_k1"),
            "g_f": _positive(g_f, "g_f"),
            "g_bna": _nonnegative(g_bna, "g_bna"),
            "g_bca": _nonnegative(g_bca, "g_bca"),
            "i_nak_max": _positive(i_nak_max, "i_nak_max"),
            "i_naca_max": _positive(i_naca_max, "i_naca_max"),
            "ca_current_scale": _positive(ca_current_scale, "ca_current_scale"),
            "uptake_max": _positive(uptake_max, "uptake_max"),
            "uptake_half_mM": _positive(uptake_half_mM, "uptake_half_mM"),
            "sr_leak_rate": _positive(sr_leak_rate, "sr_leak_rate"),
            "sr_release_rate": _positive(sr_release_rate, "sr_release_rate"),
            "sr_volume_ratio": _positive(sr_volume_ratio, "sr_volume_ratio"),
            "cytosolic_buffer_factor": _positive(
                cytosolic_buffer_factor, "cytosolic_buffer_factor"
            ),
        }
        self.phenotype = PurkinjePhenotype.HUMAN_STEWART2009_REDUCED
        for name, value in values.items():
            object.__setattr__(self, name, value)
        self.parameter_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-purkinje-stewart2009-reduced-parameters-v1",
                "phenotype": self.phenotype.value,
                "coefficients": values,
            }
        )

    def prepare(self) -> StewartPurkinjeModel:
        return StewartPurkinjeModel(self)


class PurkinjeState(StrictModule):
    """Fixed Stewart-reduced human Purkinje structure-of-arrays state."""

    voltage_mV: Array
    m: Array
    h: Array
    j: Array
    d: Array
    f: Array
    x_r: Array
    x_s: Array
    r_to: Array
    s_to: Array
    y_f: Array
    calcium_i_mM: Array
    calcium_sr_mM: Array


class PurkinjeStateRate(StrictModule):
    """Purkinje derivatives in mV/ms, gate/ms, and mM/ms."""

    voltage_mV_per_ms: Array
    m_per_ms: Array
    h_per_ms: Array
    j_per_ms: Array
    d_per_ms: Array
    f_per_ms: Array
    x_r_per_ms: Array
    x_s_per_ms: Array
    r_to_per_ms: Array
    s_to_per_ms: Array
    y_f_per_ms: Array
    calcium_i_mM_per_ms: Array
    calcium_sr_mM_per_ms: Array


class PurkinjeStateLayout(StrictModule, NonTrainableState):
    """Stable field ordering for packing the named Purkinje SoA state."""

    names: tuple[str, ...] = eqx.field(static=True)
    state_size: int = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(self):
        names = (
            "voltage_mV",
            "m",
            "h",
            "j",
            "d",
            "f",
            "x_r",
            "x_s",
            "r_to",
            "s_to",
            "y_f",
            "calcium_i_mM",
            "calcium_sr_mM",
        )
        self.names = names
        self.state_size = len(names)
        self.layout_id = canonical_fingerprint(
            {"kind": "cardiovascular-purkinje-state-layout-v1", "names": list(names)}
        )

    def index(self, name: str, /) -> int:
        if name not in self.names:
            raise KeyError(f"Unknown Purkinje state field {name!r}.")
        return self.names.index(name)

    def pack(self, state: PurkinjeState, /) -> Array:
        if not isinstance(state, PurkinjeState):
            raise TypeError("state must be PurkinjeState.")
        return jnp.stack(
            (
                state.voltage_mV,
                state.m,
                state.h,
                state.j,
                state.d,
                state.f,
                state.x_r,
                state.x_s,
                state.r_to,
                state.s_to,
                state.y_f,
                state.calcium_i_mM,
                state.calcium_sr_mM,
            ),
            axis=0,
        )

    def unpack(self, values: Array, /) -> PurkinjeState:
        array = jnp.asarray(values)
        if array.ndim == 0 or array.shape[0] != self.state_size:
            raise ValueError(
                f"Packed Purkinje state must have leading size {self.state_size}."
            )
        return PurkinjeState(*(array[index] for index in range(self.state_size)))


class PurkinjeCurrents(StrictModule):
    """Outward-positive membrane current densities in pA/pF."""

    fast_sodium: Array
    l_type_calcium: Array
    transient_outward_potassium: Array
    rapid_potassium: Array
    slow_potassium: Array
    inward_rectifier_potassium: Array
    funny: Array
    background_sodium: Array
    background_calcium: Array
    sodium_potassium_pump: Array
    sodium_calcium_exchanger: Array
    total_ionic: Array


class PurkinjeCalciumOutput(StrictModule):
    """Membrane and reduced SR calcium fluxes."""

    membrane_current_pA_per_pF: Array
    membrane_flux_mM_per_ms: Array
    uptake_flux_mM_per_ms: Array
    leak_flux_mM_per_ms: Array
    release_flux_mM_per_ms: Array
    net_cytosolic_flux_mM_per_ms: Array


class PurkinjeAdmissibilityEvidence(StrictModule):
    finite: Array
    gate_minimum: Array
    gate_maximum: Array
    maximum_gate_violation: Array
    minimum_calcium_mM: Array
    maximum_voltage_magnitude_mV: Array
    status: Array
    successful: Array


class PurkinjeRateSystem(StrictModule):
    state_rate: PurkinjeStateRate
    currents: PurkinjeCurrents
    calcium: PurkinjeCalciumOutput
    evidence: PurkinjeAdmissibilityEvidence
    gate_steady_state: Array
    gate_time_constant_ms: Array


class StewartPurkinjeModel(StrictModule, NonTrainableState):
    """Prepared Stewart-2009-informed reduced human Purkinje model."""

    parameters: StewartPurkinjeParameters
    layout: PurkinjeStateLayout
    model_id: str = eqx.field(static=True)

    def __init__(self, parameters: StewartPurkinjeParameters, /):
        if not isinstance(parameters, StewartPurkinjeParameters):
            raise TypeError("parameters must be StewartPurkinjeParameters.")
        layout = PurkinjeStateLayout()
        self.parameters = parameters
        self.layout = layout
        self.model_id = canonical_fingerprint(
            {
                "kind": "prepared-cardiovascular-purkinje-stewart2009-reduced-v1",
                "parameters": parameters.parameter_id,
                "layout": layout.layout_id,
            }
        )

    def initialize(
        self, batch_shape: Sequence[int] = (), *, dtype: jnp.dtype | None = None
    ) -> PurkinjeState:
        """Broadcast a Stewart human Purkinje resting fixture."""
        shape = _shape(batch_shape)
        resolved_dtype = jnp.asarray(0.0).dtype if dtype is None else jnp.dtype(dtype)
        values = (
            -69.137,
            0.041,
            0.1907,
            0.238,
            2.88e-4,
            0.989,
            0.045,
            0.015,
            0.001,
            0.970,
            0.10,
            1.0e-4,
            3.0,
        )
        return PurkinjeState(
            *(jnp.full(shape, value, dtype=resolved_dtype) for value in values)
        )

    def currents(self, state: PurkinjeState, /) -> PurkinjeCurrents:
        """Evaluate named outward-positive currents without hidden stimulation."""
        if not isinstance(state, PurkinjeState):
            raise TypeError("state must be PurkinjeState.")
        p = self.parameters
        v = state.voltage_mV
        ena = p.rtf_mV * jnp.log(p.sodium_o_mM / p.sodium_i_mM)
        ek = p.rtf_mV * jnp.log(p.potassium_o_mM / p.potassium_i_mM)
        eca = 0.5 * p.rtf_mV * jnp.log(p.calcium_o_mM / state.calcium_i_mM)
        i_na = p.g_na * state.m**3 * state.h * state.j * (v - ena)
        i_cal = p.g_cal * state.d * state.f * (v - eca)
        i_to = p.g_to * state.r_to * state.s_to * (v - ek)
        i_kr = p.g_kr * state.x_r * (v - ek) / (1.0 + _exp((v + 15.0) / 22.4))
        i_ks = p.g_ks * state.x_s**2 * (v - ek)
        i_k1 = p.g_k1 * (v - ek) / (1.0 + _exp(0.07 * (v + 80.0)))
        i_f = p.g_f * state.y_f * (v - p.funny_reversal_mV)
        i_bna = p.g_bna * (v - ena)
        i_bca = p.g_bca * (v - eca)
        sigma = (_exp(p.sodium_o_mM / 67.3) - 1.0) / 7.0
        f_nak = 1.0 / (
            1.0
            + 0.1245 * _exp(-0.1 * v / p.rtf_mV)
            + 0.0365 * sigma * _exp(-v / p.rtf_mV)
        )
        i_nak = (
            p.i_nak_max
            * f_nak
            * p.potassium_o_mM
            / (p.potassium_o_mM + 1.5)
            / (1.0 + (10.0 / p.sodium_i_mM) ** 1.5)
        )
        vfrt = v / p.rtf_mV
        naca_numerator = (
            _exp(0.35 * vfrt) * p.sodium_i_mM**3 * p.calcium_o_mM
            - _exp(-0.65 * vfrt) * p.sodium_o_mM**3 * state.calcium_i_mM
        )
        naca_denominator = (
            (87.5**3 + p.sodium_o_mM**3)
            * (1.38 + p.calcium_o_mM)
            * (1.0 + 0.1 * _exp(-0.65 * vfrt))
        )
        i_naca = p.i_naca_max * naca_numerator / naca_denominator
        total = (
            i_na
            + i_cal
            + i_to
            + i_kr
            + i_ks
            + i_k1
            + i_f
            + i_bna
            + i_bca
            + i_nak
            + i_naca
        )
        return PurkinjeCurrents(
            i_na,
            i_cal,
            i_to,
            i_kr,
            i_ks,
            i_k1,
            i_f,
            i_bna,
            i_bca,
            i_nak,
            i_naca,
            total,
        )

    def calcium_output(
        self, state: PurkinjeState, currents: PurkinjeCurrents, /
    ) -> PurkinjeCalciumOutput:
        if not isinstance(currents, PurkinjeCurrents):
            raise TypeError("currents must be PurkinjeCurrents.")
        p = self.parameters
        membrane_current = (
            currents.l_type_calcium
            + currents.background_calcium
            - 2.0 * currents.sodium_calcium_exchanger
        )
        membrane_flux = -p.ca_current_scale * membrane_current
        uptake = (
            p.uptake_max * state.calcium_i_mM / (p.uptake_half_mM + state.calcium_i_mM)
        )
        leak = p.sr_leak_rate * (state.calcium_sr_mM - state.calcium_i_mM)
        release = (
            p.sr_release_rate
            * state.d
            * state.f
            * (state.calcium_sr_mM - state.calcium_i_mM)
        )
        net = p.cytosolic_buffer_factor * (membrane_flux + leak + release - uptake)
        return PurkinjeCalciumOutput(
            membrane_current, membrane_flux, uptake, leak, release, net
        )

    def admissibility(self, state: PurkinjeState, /) -> PurkinjeAdmissibilityEvidence:
        packed = self.layout.pack(state)
        gates = packed[1:11]
        finite = jnp.all(jnp.isfinite(packed), axis=0)
        gate_minimum = jnp.min(gates, axis=0)
        gate_maximum = jnp.max(gates, axis=0)
        gate_violation = jnp.maximum(jnp.maximum(-gate_minimum, gate_maximum - 1.0), 0.0)
        minimum_calcium = jnp.minimum(state.calcium_i_mM, state.calcium_sr_mM)
        voltage_magnitude = jnp.abs(state.voltage_mV)
        status = jnp.zeros_like(state.voltage_mV, dtype=jnp.int32)
        status = jnp.where(
            finite, status, status | int(PurkinjeAdmissibilityStatus.NONFINITE)
        )
        status = jnp.where(
            gate_violation <= 1.0e-6,
            status,
            status | int(PurkinjeAdmissibilityStatus.GATE_OUT_OF_RANGE),
        )
        status = jnp.where(
            minimum_calcium > 0.0,
            status,
            status | int(PurkinjeAdmissibilityStatus.NONPOSITIVE_CALCIUM),
        )
        status = jnp.where(
            voltage_magnitude <= 200.0,
            status,
            status | int(PurkinjeAdmissibilityStatus.VOLTAGE_OUT_OF_RANGE),
        )
        return PurkinjeAdmissibilityEvidence(
            finite,
            gate_minimum,
            gate_maximum,
            gate_violation,
            minimum_calcium,
            voltage_magnitude,
            status,
            status == int(PurkinjeAdmissibilityStatus.SUCCESS),
        )

    def rates(
        self,
        state: PurkinjeState,
        /,
        *,
        applied_current_pA_per_pF: Array | float = 0.0,
    ) -> PurkinjeRateSystem:
        currents = self.currents(state)
        calcium = self.calcium_output(state, currents)
        evidence = self.admissibility(state)
        v = state.voltage_mV
        m_inf = jax.nn.sigmoid((v + 48.0) / 7.0)
        tau_m = 0.08 + 0.32 / (_exp((v + 40.0) / 18.0) + _exp(-(v + 40.0) / 18.0))
        h_inf = jax.nn.sigmoid(-(v + 71.0) / 6.0)
        tau_h = 0.5 + 7.0 * jax.nn.sigmoid(-(v + 40.0) / 5.0)
        j_inf = h_inf
        tau_j = 4.0 + 28.0 * jax.nn.sigmoid(-(v + 40.0) / 5.0)
        d_inf = jax.nn.sigmoid((v + 9.0) / 5.8)
        tau_d = 0.5 + 1.0 / (_exp((v + 10.0) / 30.0) + _exp(-(v + 10.0) / 30.0))
        f_inf = jax.nn.sigmoid(-(v + 25.0) / 6.0)
        tau_f = 20.0 + 80.0 * jax.nn.sigmoid(-(v + 25.0) / 5.0)
        xr_inf = jax.nn.sigmoid((v + 26.0) / 7.0)
        tau_xr = 20.0 + 100.0 / (_exp((v + 20.0) / 20.0) + _exp(-(v + 20.0) / 20.0))
        xs_inf = jax.nn.sigmoid((v - 5.0) / 14.0)
        tau_xs = 80.0 + 300.0 / (_exp((v + 20.0) / 20.0) + _exp(-(v + 20.0) / 20.0))
        r_inf = jax.nn.sigmoid((v + 10.0) / 6.0)
        tau_r = 2.0 + 6.0 / (_exp((v + 10.0) / 25.0) + _exp(-(v + 10.0) / 25.0))
        s_inf = jax.nn.sigmoid(-(v + 30.0) / 5.0)
        tau_s = 20.0 + 50.0 * jax.nn.sigmoid(-(v + 20.0) / 5.0)
        yf_inf = jax.nn.sigmoid(-(v + 80.0) / 8.0)
        tau_yf = 300.0 + 700.0 / (_exp((v + 75.0) / 25.0) + _exp(-(v + 75.0) / 25.0))
        applied = jnp.asarray(applied_current_pA_per_pF, dtype=v.dtype)
        rate = PurkinjeStateRate(
            -(currents.total_ionic + applied),
            (m_inf - state.m) / tau_m,
            (h_inf - state.h) / tau_h,
            (j_inf - state.j) / tau_j,
            (d_inf - state.d) / tau_d,
            (f_inf - state.f) / tau_f,
            (xr_inf - state.x_r) / tau_xr,
            (xs_inf - state.x_s) / tau_xs,
            (r_inf - state.r_to) / tau_r,
            (s_inf - state.s_to) / tau_s,
            (yf_inf - state.y_f) / tau_yf,
            calcium.net_cytosolic_flux_mM_per_ms,
            self.parameters.sr_volume_ratio
            * (
                calcium.uptake_flux_mM_per_ms
                - calcium.leak_flux_mM_per_ms
                - calcium.release_flux_mM_per_ms
            ),
        )
        gate_steady_state = jnp.stack(
            (
                m_inf,
                h_inf,
                j_inf,
                d_inf,
                f_inf,
                xr_inf,
                xs_inf,
                r_inf,
                s_inf,
                yf_inf,
            ),
            axis=-1,
        )
        gate_time_constant_ms = jnp.stack(
            (
                tau_m,
                tau_h,
                tau_j,
                tau_d,
                tau_f,
                tau_xr,
                tau_xs,
                tau_r,
                tau_s,
                tau_yf,
            ),
            axis=-1,
        )
        return PurkinjeRateSystem(
            rate,
            currents,
            calcium,
            evidence,
            gate_steady_state,
            gate_time_constant_ms,
        )


_PURKINJE_REACTION_PARAMETER_NAMES = (
    "rtf_mV",
    "sodium_i_mM",
    "sodium_o_mM",
    "potassium_i_mM",
    "potassium_o_mM",
    "calcium_o_mM",
    "funny_reversal_mV",
    "g_na",
    "g_cal",
    "g_to",
    "g_kr",
    "g_ks",
    "g_k1",
    "g_f",
    "g_bna",
    "g_bca",
    "i_nak_max",
    "i_naca_max",
    "ca_current_scale",
    "uptake_max",
    "uptake_half_mM",
    "sr_leak_rate",
    "sr_release_rate",
    "sr_volume_ratio",
    "cytosolic_buffer_factor",
)


def _purkinje_parameter_values(
    parameters: StewartPurkinjeParameters, /
) -> tuple[float, ...]:
    return (
        parameters.rtf_mV,
        parameters.sodium_i_mM,
        parameters.sodium_o_mM,
        parameters.potassium_i_mM,
        parameters.potassium_o_mM,
        parameters.calcium_o_mM,
        parameters.funny_reversal_mV,
        parameters.g_na,
        parameters.g_cal,
        parameters.g_to,
        parameters.g_kr,
        parameters.g_ks,
        parameters.g_k1,
        parameters.g_f,
        parameters.g_bna,
        parameters.g_bca,
        parameters.i_nak_max,
        parameters.i_naca_max,
        parameters.ca_current_scale,
        parameters.uptake_max,
        parameters.uptake_half_mM,
        parameters.sr_leak_rate,
        parameters.sr_release_rate,
        parameters.sr_volume_ratio,
        parameters.cytosolic_buffer_factor,
    )


def _purkinje_reaction_state(state: PurkinjeState, /) -> Array:
    return jnp.stack(
        (
            state.voltage_mV,
            state.m,
            state.h,
            state.j,
            state.d,
            state.f,
            state.x_r,
            state.x_s,
            state.r_to,
            state.s_to,
            state.y_f,
            state.calcium_i_mM,
            state.calcium_sr_mM,
        ),
        axis=-1,
    )


def _purkinje_native_state(state: Array, /) -> PurkinjeState:
    return PurkinjeState(*(state[..., index] for index in range(13)))


def _purkinje_reaction_rate(rate: PurkinjeStateRate, /) -> Array:
    return jnp.stack(
        (
            rate.voltage_mV_per_ms,
            rate.m_per_ms,
            rate.h_per_ms,
            rate.j_per_ms,
            rate.d_per_ms,
            rate.f_per_ms,
            rate.x_r_per_ms,
            rate.x_s_per_ms,
            rate.r_to_per_ms,
            rate.s_to_per_ms,
            rate.y_f_per_ms,
            rate.calcium_i_mM_per_ms,
            rate.calcium_sr_mM_per_ms,
        ),
        axis=-1,
    )


def _purkinje_reaction_currents(currents: PurkinjeCurrents, /) -> Array:
    return jnp.stack(
        (
            currents.fast_sodium,
            currents.l_type_calcium,
            currents.transient_outward_potassium,
            currents.rapid_potassium,
            currents.slow_potassium,
            currents.inward_rectifier_potassium,
            currents.funny,
            currents.background_sodium,
            currents.background_calcium,
            currents.sodium_potassium_pump,
            currents.sodium_calcium_exchanger,
        ),
        axis=-1,
    )


@dataclass(frozen=True)
class StewartPurkinjeReactionAdapter:
    """Final-axis reaction adapter for one homogeneous typed Purkinje model."""

    cell_model: StewartPurkinjeModel = field(
        default_factory=lambda: StewartPurkinjeParameters().prepare()
    )
    scaling: CardiacMembraneScaling = field(default_factory=CardiacMembraneScaling)
    model_id: str = field(init=False)
    default_parameters: Array = field(init=False, repr=False, compare=False)

    state_layout: ClassVar[CardiacReactionStateLayout] = CardiacReactionStateLayout(
        PurkinjeStateLayout().names,
        ("mV",) + ("1",) * 10 + ("mM", "mM"),
        PurkinjeStateLayout().names[1:11],
        PurkinjeStateLayout().names[11:],
    )
    parameter_layout: ClassVar[CardiacReactionParameterLayout] = (
        CardiacReactionParameterLayout(
            _PURKINJE_REACTION_PARAMETER_NAMES,
            (
                "mV",
                "mM",
                "mM",
                "mM",
                "mM",
                "mM",
                "mV",
                "nS/pF",
                "nS/pF",
                "nS/pF",
                "nS/pF",
                "nS/pF",
                "nS/pF",
                "nS/pF",
                "nS/pF",
                "nS/pF",
                "pA/pF",
                "pA/pF",
                "mM ms^-1/(pA pF^-1)",
                "mM/ms",
                "mM",
                "1/ms",
                "1/ms",
                "1",
                "1",
            ),
        )
    )
    current_names: ClassVar[tuple[str, ...]] = (
        "I_Na",
        "I_CaL",
        "I_to",
        "I_Kr",
        "I_Ks",
        "I_K1",
        "I_f",
        "I_bNa",
        "I_bCa",
        "I_NaK",
        "I_NaCa",
    )

    def __post_init__(self) -> None:
        if not isinstance(self.cell_model, StewartPurkinjeModel):
            raise TypeError("cell_model must be StewartPurkinjeModel.")
        if not isinstance(self.scaling, CardiacMembraneScaling):
            raise TypeError("scaling must be CardiacMembraneScaling.")
        object.__setattr__(
            self,
            "default_parameters",
            jnp.asarray(_purkinje_parameter_values(self.cell_model.parameters)),
        )
        object.__setattr__(
            self,
            "model_id",
            canonical_fingerprint(
                {
                    "kind": "cardiovascular-purkinje-reaction-adapter-v1",
                    "cell_model": self.cell_model.model_id,
                    "membrane_surface_to_volume_per_mm": (
                        self.scaling.membrane_surface_to_volume_per_mm
                    ),
                    "membrane_capacitance_uF_per_mm2": (
                        self.scaling.membrane_capacitance_uF_per_mm2
                    ),
                }
            ),
        )

    @property
    def membrane_capacitance_uF_per_mm2(self) -> float:
        return self.scaling.membrane_capacitance_uF_per_mm2

    @property
    def membrane_surface_to_volume_per_mm(self) -> float:
        return self.scaling.membrane_surface_to_volume_per_mm

    def _parameters(self, parameters: Array | None, dtype: object) -> Array:
        if parameters is None:
            return jnp.asarray(self.default_parameters, dtype=dtype)
        return self.parameter_layout.require_shape(parameters).astype(dtype)

    def _parameter_admissible(self, parameters: Array) -> Array:
        expected = jnp.asarray(self.default_parameters, dtype=parameters.dtype)
        return jnp.all(jnp.isfinite(parameters) & (parameters == expected), axis=-1)

    def initialize(
        self,
        batch_shape: tuple[int, ...] = (),
        *,
        dtype: object | None = None,
    ) -> Array:
        return _purkinje_reaction_state(
            self.cell_model.initialize(batch_shape, dtype=dtype)
        )

    def admissible(self, state: Array, parameters: Array | None = None) -> Array:
        resolved = self.state_layout.require_shape(state)
        configured = self._parameters(parameters, resolved.dtype)
        return self.cell_model.admissibility(
            _purkinje_native_state(resolved)
        ).successful & self._parameter_admissible(configured)

    def evaluate(
        self,
        state: Array,
        parameters: Array | None = None,
        *,
        stimulus_current_uA_per_mm2: ArrayLike = 0.0,
    ) -> CardiacReactionEvaluation:
        resolved = self.state_layout.require_shape(state)
        configured = self._parameters(parameters, resolved.dtype)
        native = _purkinje_native_state(resolved)
        stimulus = jnp.asarray(stimulus_current_uA_per_mm2, dtype=resolved.dtype)
        system = self.cell_model.rates(
            native,
            applied_current_pA_per_pF=(stimulus / self.membrane_capacitance_uF_per_mm2),
        )
        state_rate = _purkinje_reaction_rate(system.state_rate)
        current_density = (
            _purkinje_reaction_currents(system.currents)
            * self.membrane_capacitance_uF_per_mm2
        )
        total_current = jnp.sum(current_density, axis=-1)
        valid = system.evidence.successful & self._parameter_admissible(configured)
        nan = jnp.asarray(jnp.nan, dtype=resolved.dtype)
        sr_flux = (
            system.calcium.release_flux_mM_per_ms
            + system.calcium.leak_flux_mM_per_ms
            - system.calcium.uptake_flux_mM_per_ms
        )
        charge_residual = (
            self.membrane_capacitance_uF_per_mm2 * state_rate[..., 0]
            + total_current
            + stimulus
        )
        return CardiacReactionEvaluation(
            state_rate=jnp.where(valid[..., None], state_rate, nan),
            gate_steady_state=jnp.where(valid[..., None], system.gate_steady_state, nan),
            gate_time_constant_ms=jnp.where(
                valid[..., None], system.gate_time_constant_ms, nan
            ),
            current_density_uA_per_mm2=jnp.where(valid[..., None], current_density, nan),
            total_outward_current_uA_per_mm2=jnp.where(valid, total_current, nan),
            calcium_cytosol_mM=jnp.where(valid, native.calcium_i_mM, nan),
            calcium_cytosol_rate_mM_per_ms=jnp.where(
                valid, system.state_rate.calcium_i_mM_per_ms, nan
            ),
            calcium_sr_flux_mM_per_ms=jnp.where(valid, sr_flux, nan),
            calcium_membrane_current_uA_per_mm2=jnp.where(
                valid,
                system.calcium.membrane_current_pA_per_pF
                * self.membrane_capacitance_uF_per_mm2,
                nan,
            ),
            charge_balance_residual_uA_per_mm2=jnp.where(valid, charge_residual, nan),
            valid=valid,
            current_names=self.current_names,
            model_id=self.model_id,
        )

    def rates(
        self,
        state: Array,
        parameters: Array | None = None,
        *,
        stimulus_current_uA_per_mm2: ArrayLike = 0.0,
    ) -> Array:
        return self.evaluate(
            state,
            parameters,
            stimulus_current_uA_per_mm2=stimulus_current_uA_per_mm2,
        ).state_rate

    def currents(self, state: Array, parameters: Array | None = None) -> Array:
        return self.evaluate(state, parameters).current_density_uA_per_mm2

    def exact_gate_update(
        self,
        state: Array,
        dt_ms: ArrayLike,
        parameters: Array | None = None,
    ) -> Array:
        resolved = self.state_layout.require_shape(state)
        evaluation = self.evaluate(resolved, parameters)
        dt = jnp.asarray(dt_ms, dtype=resolved.dtype)
        gate_indices = jnp.asarray(self.state_layout.gate_indices)
        old_gates = resolved[..., gate_indices]
        updated_gates = evaluation.gate_steady_state + (
            old_gates - evaluation.gate_steady_state
        ) * jnp.exp(-dt[..., None] / evaluation.gate_time_constant_ms)
        updated = resolved.at[..., gate_indices].set(updated_gates)
        valid = evaluation.valid & jnp.isfinite(dt) & (dt >= 0.0)
        return jnp.where(valid[..., None], updated, jnp.nan)

    def validate_state(
        self, state: ArrayLike, parameters: ArrayLike | None = None
    ) -> None:
        array = np.asarray(state)
        if array.ndim == 0 or array.shape[-1] != self.state_layout.state_count:
            raise ValueError(
                "Purkinje reaction state must have final axis size "
                f"{self.state_layout.state_count}, received {array.shape}."
            )
        parameter_array = None if parameters is None else jnp.asarray(parameters)
        if not np.all(np.isfinite(array)) or not np.all(
            np.asarray(self.admissible(jnp.asarray(array), parameter_array))
        ):
            raise ValueError(
                "Purkinje reaction state or configured parameters are inadmissible."
            )


__all__ = [
    "PurkinjeAdmissibilityEvidence",
    "PurkinjeAdmissibilityStatus",
    "PurkinjeCalciumOutput",
    "PurkinjeCurrents",
    "PurkinjePhenotype",
    "PurkinjeRateSystem",
    "PurkinjeState",
    "PurkinjeStateLayout",
    "PurkinjeStateRate",
    "StewartPurkinjeModel",
    "StewartPurkinjeParameters",
    "StewartPurkinjeReactionAdapter",
]
