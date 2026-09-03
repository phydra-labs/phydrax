#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Qualified reduced human atrial working-myocyte ionic dynamics.

The equations retain the phenotype-defining Courtemanche--Ramirez--Nattel
fast sodium, atrial ``I_to``/``I_Kur``, delayed-rectifier, L-type calcium,
pump, exchanger, and one-pool SR mechanisms.  This is intentionally a
compact membrane/Ca subsystem, not a claim to reproduce the full 21-state
1998 formulation.
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


class AtrialPhenotype(Enum):
    """Qualified atrial phenotype identity; never a boolean model switch."""

    HUMAN_WORKING_MYOCYTE_CRN1998_REDUCED = "human-working-myocyte-crn1998-reduced"


class AtrialAdmissibilityStatus(IntFlag):
    """Fail-closed state admissibility bits."""

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


def _exp(value: Array, /) -> Array:
    return jnp.exp(jnp.clip(value, -80.0, 80.0))


def _x_over_expm1(x: Array, scale: float, /) -> Array:
    z = x / scale
    small = jnp.abs(z) < 1.0e-6
    denominator = jnp.where(small, jnp.ones_like(z), jnp.expm1(z))
    regular = x / denominator
    series = scale * (1.0 - 0.5 * z + z * z / 12.0)
    return jnp.where(small, series, regular)


def _x_over_one_minus_exp_negative(x: Array, scale: float, /) -> Array:
    return _x_over_expm1(-x, scale)


def _broadcast(value: float, batch_shape: tuple[int, ...], dtype: jnp.dtype, /) -> Array:
    return jnp.full(batch_shape, value, dtype=dtype)


def _batch_shape(value: Sequence[int], /) -> tuple[int, ...]:
    shape = tuple(value)
    if any(
        isinstance(axis, bool) or not isinstance(axis, int) or axis < 0 for axis in shape
    ):
        raise ValueError("batch_shape axes must be nonnegative integers.")
    return shape


class CourtemancheAtrialParameters(StrictModule, NonTrainableState):
    """Host-side coefficient plan for the qualified atrial subsystem."""

    phenotype: AtrialPhenotype = eqx.field(static=True)
    rtf_mV: float = eqx.field(static=True)
    faraday_C_per_mmol: float = eqx.field(static=True)
    sodium_i_mM: float = eqx.field(static=True)
    sodium_o_mM: float = eqx.field(static=True)
    potassium_i_mM: float = eqx.field(static=True)
    potassium_o_mM: float = eqx.field(static=True)
    calcium_o_mM: float = eqx.field(static=True)
    g_na: float = eqx.field(static=True)
    g_k1: float = eqx.field(static=True)
    g_to: float = eqx.field(static=True)
    g_kur_scale: float = eqx.field(static=True)
    g_kr: float = eqx.field(static=True)
    g_ks: float = eqx.field(static=True)
    g_cal: float = eqx.field(static=True)
    g_bna: float = eqx.field(static=True)
    g_bca: float = eqx.field(static=True)
    i_nak_max: float = eqx.field(static=True)
    i_naca_max: float = eqx.field(static=True)
    i_cap_max: float = eqx.field(static=True)
    q10_k: float = eqx.field(static=True)
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
        faraday_C_per_mmol: float = 96.4867,
        sodium_i_mM: float = 11.17,
        sodium_o_mM: float = 140.0,
        potassium_i_mM: float = 139.0,
        potassium_o_mM: float = 5.4,
        calcium_o_mM: float = 1.8,
        g_na: float = 7.8,
        g_k1: float = 0.09,
        g_to: float = 0.1652,
        g_kur_scale: float = 1.0,
        g_kr: float = 0.029411765,
        g_ks: float = 0.12941176,
        g_cal: float = 0.12375,
        g_bna: float = 0.0006744375,
        g_bca: float = 0.001131,
        i_nak_max: float = 0.59933874,
        i_naca_max: float = 1600.0,
        i_cap_max: float = 0.275,
        q10_k: float = 3.0,
        ca_current_scale: float = 3.791e-5,
        uptake_max: float = 0.005,
        uptake_half_mM: float = 0.00092,
        sr_leak_rate: float = 3.0e-4,
        sr_release_rate: float = 0.30,
        sr_volume_ratio: float = 13.0,
        cytosolic_buffer_factor: float = 0.10,
    ):
        values = {
            "rtf_mV": _positive(rtf_mV, "rtf_mV"),
            "faraday_C_per_mmol": _positive(faraday_C_per_mmol, "faraday_C_per_mmol"),
            "sodium_i_mM": _positive(sodium_i_mM, "sodium_i_mM"),
            "sodium_o_mM": _positive(sodium_o_mM, "sodium_o_mM"),
            "potassium_i_mM": _positive(potassium_i_mM, "potassium_i_mM"),
            "potassium_o_mM": _positive(potassium_o_mM, "potassium_o_mM"),
            "calcium_o_mM": _positive(calcium_o_mM, "calcium_o_mM"),
            "g_na": _positive(g_na, "g_na"),
            "g_k1": _positive(g_k1, "g_k1"),
            "g_to": _positive(g_to, "g_to"),
            "g_kur_scale": _positive(g_kur_scale, "g_kur_scale"),
            "g_kr": _positive(g_kr, "g_kr"),
            "g_ks": _positive(g_ks, "g_ks"),
            "g_cal": _positive(g_cal, "g_cal"),
            "g_bna": _nonnegative(g_bna, "g_bna"),
            "g_bca": _nonnegative(g_bca, "g_bca"),
            "i_nak_max": _positive(i_nak_max, "i_nak_max"),
            "i_naca_max": _positive(i_naca_max, "i_naca_max"),
            "i_cap_max": _positive(i_cap_max, "i_cap_max"),
            "q10_k": _positive(q10_k, "q10_k"),
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
        self.phenotype = AtrialPhenotype.HUMAN_WORKING_MYOCYTE_CRN1998_REDUCED
        for name, value in values.items():
            object.__setattr__(self, name, value)
        self.parameter_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-atrial-crn1998-reduced-parameters-v1",
                "phenotype": self.phenotype.value,
                "coefficients": values,
            }
        )

    def prepare(self) -> CourtemancheAtrialModel:
        """Freeze the coefficient plan into a model/runtime identity."""
        return CourtemancheAtrialModel(self)


class AtrialState(StrictModule):
    """Fixed named structure-of-arrays state; every field has one batch shape."""

    voltage_mV: Array
    m: Array
    h: Array
    j: Array
    oa: Array
    oi: Array
    ua: Array
    ui: Array
    xr: Array
    xs: Array
    d: Array
    f: Array
    f_ca: Array
    calcium_i_mM: Array
    calcium_sr_mM: Array


class AtrialStateRate(StrictModule):
    """Time derivatives in mV/ms, gate/ms, and mM/ms."""

    voltage_mV_per_ms: Array
    m_per_ms: Array
    h_per_ms: Array
    j_per_ms: Array
    oa_per_ms: Array
    oi_per_ms: Array
    ua_per_ms: Array
    ui_per_ms: Array
    xr_per_ms: Array
    xs_per_ms: Array
    d_per_ms: Array
    f_per_ms: Array
    f_ca_per_ms: Array
    calcium_i_mM_per_ms: Array
    calcium_sr_mM_per_ms: Array


class AtrialStateLayout(StrictModule, NonTrainableState):
    """Stable field ordering for packing the named SoA state."""

    names: tuple[str, ...] = eqx.field(static=True)
    state_size: int = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(self):
        names = (
            "voltage_mV",
            "m",
            "h",
            "j",
            "oa",
            "oi",
            "ua",
            "ui",
            "xr",
            "xs",
            "d",
            "f",
            "f_ca",
            "calcium_i_mM",
            "calcium_sr_mM",
        )
        self.names = names
        self.state_size = len(names)
        self.layout_id = canonical_fingerprint(
            {"kind": "cardiovascular-atrial-state-layout-v1", "names": list(names)}
        )

    def index(self, name: str, /) -> int:
        if name not in self.names:
            raise KeyError(f"Unknown atrial state field {name!r}.")
        return self.names.index(name)

    def pack(self, state: AtrialState, /) -> Array:
        if not isinstance(state, AtrialState):
            raise TypeError("state must be an AtrialState.")
        return jnp.stack(
            (
                state.voltage_mV,
                state.m,
                state.h,
                state.j,
                state.oa,
                state.oi,
                state.ua,
                state.ui,
                state.xr,
                state.xs,
                state.d,
                state.f,
                state.f_ca,
                state.calcium_i_mM,
                state.calcium_sr_mM,
            ),
            axis=0,
        )

    def unpack(self, values: Array, /) -> AtrialState:
        array = jnp.asarray(values)
        if array.ndim == 0 or array.shape[0] != self.state_size:
            raise ValueError(
                f"Packed atrial state must have leading size {self.state_size}."
            )
        return AtrialState(*(array[index] for index in range(self.state_size)))


class AtrialCurrents(StrictModule):
    """Outward-positive membrane current densities in pA/pF."""

    fast_sodium: Array
    inward_rectifier_potassium: Array
    transient_outward_potassium: Array
    ultrarapid_potassium: Array
    rapid_potassium: Array
    slow_potassium: Array
    l_type_calcium: Array
    background_sodium: Array
    background_calcium: Array
    sodium_potassium_pump: Array
    sodium_calcium_exchanger: Array
    sarcolemmal_calcium_pump: Array
    total_ionic: Array


class AtrialCalciumOutput(StrictModule):
    """Membrane and SR calcium-flux evidence in the model's native units."""

    membrane_current_pA_per_pF: Array
    membrane_flux_mM_per_ms: Array
    uptake_flux_mM_per_ms: Array
    leak_flux_mM_per_ms: Array
    release_flux_mM_per_ms: Array
    net_cytosolic_flux_mM_per_ms: Array


class AtrialAdmissibilityEvidence(StrictModule):
    """Per-cell finite, gate, calcium, and voltage admissibility evidence."""

    finite: Array
    gate_minimum: Array
    gate_maximum: Array
    maximum_gate_violation: Array
    minimum_calcium_mM: Array
    maximum_voltage_magnitude_mV: Array
    status: Array
    successful: Array


class AtrialRateSystem(StrictModule):
    """One evaluation of the coupled atrial governing rate/current system."""

    state_rate: AtrialStateRate
    currents: AtrialCurrents
    calcium: AtrialCalciumOutput
    evidence: AtrialAdmissibilityEvidence
    gate_steady_state: Array
    gate_time_constant_ms: Array


class CourtemancheAtrialModel(StrictModule, NonTrainableState):
    """Prepared qualified CRN-1998 reduced human atrial model."""

    parameters: CourtemancheAtrialParameters
    layout: AtrialStateLayout
    model_id: str = eqx.field(static=True)

    def __init__(self, parameters: CourtemancheAtrialParameters, /):
        if not isinstance(parameters, CourtemancheAtrialParameters):
            raise TypeError("parameters must be CourtemancheAtrialParameters.")
        layout = AtrialStateLayout()
        self.parameters = parameters
        self.layout = layout
        self.model_id = canonical_fingerprint(
            {
                "kind": "prepared-cardiovascular-atrial-crn1998-reduced-v1",
                "parameters": parameters.parameter_id,
                "layout": layout.layout_id,
            }
        )

    def initialize(
        self, batch_shape: Sequence[int] = (), *, dtype: jnp.dtype | None = None
    ) -> AtrialState:
        """Broadcast the published CRN resting fixture to a fixed batch shape."""
        shape = _batch_shape(batch_shape)
        resolved_dtype = jnp.asarray(0.0).dtype if dtype is None else jnp.dtype(dtype)
        values = (
            -81.18,
            2.908e-3,
            9.649e-1,
            9.775e-1,
            3.043e-2,
            9.992e-1,
            4.966e-3,
            9.986e-1,
            3.296e-5,
            1.869e-2,
            1.367e-4,
            9.996e-1,
            7.755e-1,
            1.013e-4,
            1.488,
        )
        return AtrialState(
            *(_broadcast(value, shape, resolved_dtype) for value in values)
        )

    def currents(self, state: AtrialState, /) -> AtrialCurrents:
        """Evaluate named outward-positive currents without hidden stimulation."""
        if not isinstance(state, AtrialState):
            raise TypeError("state must be an AtrialState.")
        p = self.parameters
        v = state.voltage_mV
        ena = p.rtf_mV * jnp.log(p.sodium_o_mM / p.sodium_i_mM)
        ek = p.rtf_mV * jnp.log(p.potassium_o_mM / p.potassium_i_mM)
        eca = 0.5 * p.rtf_mV * jnp.log(p.calcium_o_mM / state.calcium_i_mM)
        i_na = p.g_na * state.m**3 * state.h * state.j * (v - ena)
        i_k1 = p.g_k1 * (v - ek) / (1.0 + _exp(0.07 * (v + 80.0)))
        i_to = p.g_to * state.oa**3 * state.oi * (v - ek)
        g_kur = p.g_kur_scale * (0.005 + 0.05 * jax.nn.sigmoid((v - 15.0) / 13.0))
        i_kur = g_kur * state.ua**3 * state.ui * (v - ek)
        i_kr = p.g_kr * state.xr * (v - ek) / (1.0 + _exp((v + 15.0) / 22.4))
        i_ks = p.g_ks * state.xs**2 * (v - ek)
        i_cal = p.g_cal * state.d * state.f * state.f_ca * (v - 65.0)
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
        i_cap = p.i_cap_max * state.calcium_i_mM / (0.0005 + state.calcium_i_mM)
        total = (
            i_na
            + i_k1
            + i_to
            + i_kur
            + i_kr
            + i_ks
            + i_cal
            + i_bna
            + i_bca
            + i_nak
            + i_naca
            + i_cap
        )
        return AtrialCurrents(
            i_na,
            i_k1,
            i_to,
            i_kur,
            i_kr,
            i_ks,
            i_cal,
            i_bna,
            i_bca,
            i_nak,
            i_naca,
            i_cap,
            total,
        )

    def calcium_output(
        self, state: AtrialState, currents: AtrialCurrents, /
    ) -> AtrialCalciumOutput:
        """Resolve current-to-concentration and reduced one-pool SR fluxes."""
        if not isinstance(state, AtrialState):
            raise TypeError("state must be an AtrialState.")
        if not isinstance(currents, AtrialCurrents):
            raise TypeError("currents must be AtrialCurrents.")
        p = self.parameters
        membrane_current = (
            currents.l_type_calcium
            + currents.background_calcium
            + currents.sarcolemmal_calcium_pump
            - 2.0 * currents.sodium_calcium_exchanger
        )
        current_to_concentration = p.ca_current_scale * 96.4867 / p.faraday_C_per_mmol
        membrane_flux = -current_to_concentration * membrane_current
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
        return AtrialCalciumOutput(
            membrane_current, membrane_flux, uptake, leak, release, net
        )

    def admissibility(self, state: AtrialState, /) -> AtrialAdmissibilityEvidence:
        """Return fail-closed array evidence without host synchronization."""
        packed = self.layout.pack(state)
        gates = packed[1:13]
        finite = jnp.all(jnp.isfinite(packed), axis=0)
        gate_minimum = jnp.min(gates, axis=0)
        gate_maximum = jnp.max(gates, axis=0)
        gate_violation = jnp.maximum(jnp.maximum(-gate_minimum, gate_maximum - 1.0), 0.0)
        minimum_calcium = jnp.minimum(state.calcium_i_mM, state.calcium_sr_mM)
        voltage_magnitude = jnp.abs(state.voltage_mV)
        status = jnp.zeros_like(state.voltage_mV, dtype=jnp.int32)
        status = jnp.where(
            finite, status, status | int(AtrialAdmissibilityStatus.NONFINITE)
        )
        status = jnp.where(
            gate_violation <= 1.0e-6,
            status,
            status | int(AtrialAdmissibilityStatus.GATE_OUT_OF_RANGE),
        )
        status = jnp.where(
            minimum_calcium > 0.0,
            status,
            status | int(AtrialAdmissibilityStatus.NONPOSITIVE_CALCIUM),
        )
        status = jnp.where(
            voltage_magnitude <= 200.0,
            status,
            status | int(AtrialAdmissibilityStatus.VOLTAGE_OUT_OF_RANGE),
        )
        return AtrialAdmissibilityEvidence(
            finite,
            gate_minimum,
            gate_maximum,
            gate_violation,
            minimum_calcium,
            voltage_magnitude,
            status,
            status == int(AtrialAdmissibilityStatus.SUCCESS),
        )

    def rates(
        self,
        state: AtrialState,
        /,
        *,
        applied_current_pA_per_pF: Array | float = 0.0,
    ) -> AtrialRateSystem:
        """Evaluate the coupled ODE; applied current follows outward-positive sign."""
        currents = self.currents(state)
        calcium = self.calcium_output(state, currents)
        evidence = self.admissibility(state)
        v = state.voltage_mV
        alpha_m = 0.32 * _x_over_one_minus_exp_negative(v + 47.13, 10.0)
        beta_m = 0.08 * _exp(-v / 11.0)
        m_inf = alpha_m / (alpha_m + beta_m)
        tau_m = 1.0 / (alpha_m + beta_m)
        alpha_h = jnp.where(v < -40.0, 0.135 * _exp(-(v + 80.0) / 6.8), 0.0)
        beta_h = jnp.where(
            v < -40.0,
            3.56 * _exp(0.079 * v) + 3.1e5 * _exp(0.35 * v),
            1.0 / (0.13 * (1.0 + _exp(-(v + 10.66) / 11.1))),
        )
        h_inf = alpha_h / (alpha_h + beta_h)
        tau_h = 1.0 / (alpha_h + beta_h)
        alpha_j = jnp.where(
            v < -40.0,
            (-1.2714e5 * _exp(0.2444 * v) - 3.474e-5 * _exp(-0.04391 * v))
            * (v + 37.78)
            / (1.0 + _exp(0.311 * (v + 79.23))),
            0.0,
        )
        beta_j = jnp.where(
            v < -40.0,
            0.1212 * _exp(-0.01052 * v) / (1.0 + _exp(-0.1378 * (v + 40.14))),
            0.3 * _exp(-2.535e-7 * v) / (1.0 + _exp(-0.1 * (v + 32.0))),
        )
        j_inf = alpha_j / (alpha_j + beta_j)
        tau_j = 1.0 / (alpha_j + beta_j)
        alpha_oa = 0.65 / (_exp(-(v + 10.0) / 8.5) + _exp(-(v - 30.0) / 59.0))
        beta_oa = 0.65 / (2.5 + _exp((v + 82.0) / 17.0))
        tau_oa = 1.0 / (self.parameters.q10_k * (alpha_oa + beta_oa))
        oa_inf = jax.nn.sigmoid((v + 20.47) / 17.54)
        alpha_oi = 1.0 / (18.53 + _exp((v + 113.7) / 10.95))
        beta_oi = 1.0 / (35.56 + _exp(-(v + 1.26) / 7.44))
        tau_oi = 1.0 / (self.parameters.q10_k * (alpha_oi + beta_oi))
        oi_inf = jax.nn.sigmoid(-(v + 43.1) / 5.3)
        ua_inf = jax.nn.sigmoid((v + 30.3) / 9.6)
        tau_ua = tau_oa
        alpha_ui = 1.0 / (21.0 + _exp(-(v - 185.0) / 28.0))
        beta_ui = _exp((v - 158.0) / 16.0)
        tau_ui = 1.0 / (self.parameters.q10_k * (alpha_ui + beta_ui))
        ui_inf = jax.nn.sigmoid(-(v - 99.45) / 27.48)
        alpha_xr = 0.0003 * _x_over_one_minus_exp_negative(v + 14.1, 5.0)
        beta_xr = 7.3898e-5 * _x_over_expm1(v - 3.3328, 5.1237)
        xr_inf = jax.nn.sigmoid((v + 14.1) / 6.5)
        tau_xr = 1.0 / (alpha_xr + beta_xr)
        alpha_xs = 4.0e-5 * _x_over_one_minus_exp_negative(v - 19.9, 17.0)
        beta_xs = 3.5e-5 * _x_over_expm1(v - 19.9, 9.0)
        xs_inf = jnp.sqrt(jax.nn.sigmoid((v - 19.9) / 12.7))
        tau_xs = 0.5 / (alpha_xs + beta_xs)
        d_inf = jax.nn.sigmoid((v + 10.0) / 8.0)
        d_tau_denominator = 0.035 * (v + 10.0) * (1.0 + _exp(-(v + 10.0) / 6.24))
        d_regular = (1.0 - _exp(-(v + 10.0) / 6.24)) / jnp.where(
            jnp.abs(v + 10.0) < 1.0e-6, jnp.ones_like(v), d_tau_denominator
        )
        tau_d = jnp.where(
            jnp.abs(v + 10.0) < 1.0e-6,
            4.579 / (1.0 + _exp(-(v + 10.0) / 6.24)),
            d_regular,
        )
        f_inf = jax.nn.sigmoid(-(v + 28.0) / 6.9)
        tau_f = 9.0 / (0.0197 * _exp(-(0.0337**2) * (v + 10.0) ** 2) + 0.02)
        f_ca_inf = 1.0 / (1.0 + state.calcium_i_mM / 0.00035)
        applied = jnp.asarray(applied_current_pA_per_pF, dtype=v.dtype)
        rate = AtrialStateRate(
            -(currents.total_ionic + applied),
            (m_inf - state.m) / tau_m,
            (h_inf - state.h) / tau_h,
            (j_inf - state.j) / tau_j,
            (oa_inf - state.oa) / tau_oa,
            (oi_inf - state.oi) / tau_oi,
            (ua_inf - state.ua) / tau_ua,
            (ui_inf - state.ui) / tau_ui,
            (xr_inf - state.xr) / tau_xr,
            (xs_inf - state.xs) / tau_xs,
            (d_inf - state.d) / tau_d,
            (f_inf - state.f) / tau_f,
            (f_ca_inf - state.f_ca) / 2.0,
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
                oa_inf,
                oi_inf,
                ua_inf,
                ui_inf,
                xr_inf,
                xs_inf,
                d_inf,
                f_inf,
                f_ca_inf,
            ),
            axis=-1,
        )
        gate_time_constant_ms = jnp.stack(
            (
                tau_m,
                tau_h,
                tau_j,
                tau_oa,
                tau_oi,
                tau_ua,
                tau_ui,
                tau_xr,
                tau_xs,
                tau_d,
                tau_f,
                jnp.full_like(v, 2.0),
            ),
            axis=-1,
        )
        return AtrialRateSystem(
            rate,
            currents,
            calcium,
            evidence,
            gate_steady_state,
            gate_time_constant_ms,
        )


_ATRIAL_REACTION_PARAMETER_NAMES = (
    "rtf_mV",
    "faraday_C_per_mmol",
    "sodium_i_mM",
    "sodium_o_mM",
    "potassium_i_mM",
    "potassium_o_mM",
    "calcium_o_mM",
    "g_na",
    "g_k1",
    "g_to",
    "g_kur_scale",
    "g_kr",
    "g_ks",
    "g_cal",
    "g_bna",
    "g_bca",
    "i_nak_max",
    "i_naca_max",
    "i_cap_max",
    "q10_k",
    "ca_current_scale",
    "uptake_max",
    "uptake_half_mM",
    "sr_leak_rate",
    "sr_release_rate",
    "sr_volume_ratio",
    "cytosolic_buffer_factor",
)


def _atrial_parameter_values(
    parameters: CourtemancheAtrialParameters, /
) -> tuple[float, ...]:
    return (
        parameters.rtf_mV,
        parameters.faraday_C_per_mmol,
        parameters.sodium_i_mM,
        parameters.sodium_o_mM,
        parameters.potassium_i_mM,
        parameters.potassium_o_mM,
        parameters.calcium_o_mM,
        parameters.g_na,
        parameters.g_k1,
        parameters.g_to,
        parameters.g_kur_scale,
        parameters.g_kr,
        parameters.g_ks,
        parameters.g_cal,
        parameters.g_bna,
        parameters.g_bca,
        parameters.i_nak_max,
        parameters.i_naca_max,
        parameters.i_cap_max,
        parameters.q10_k,
        parameters.ca_current_scale,
        parameters.uptake_max,
        parameters.uptake_half_mM,
        parameters.sr_leak_rate,
        parameters.sr_release_rate,
        parameters.sr_volume_ratio,
        parameters.cytosolic_buffer_factor,
    )


def _atrial_reaction_state(state: AtrialState, /) -> Array:
    return jnp.stack(
        (
            state.voltage_mV,
            state.m,
            state.h,
            state.j,
            state.oa,
            state.oi,
            state.ua,
            state.ui,
            state.xr,
            state.xs,
            state.d,
            state.f,
            state.f_ca,
            state.calcium_i_mM,
            state.calcium_sr_mM,
        ),
        axis=-1,
    )


def _atrial_native_state(state: Array, /) -> AtrialState:
    return AtrialState(*(state[..., index] for index in range(15)))


def _atrial_reaction_rate(rate: AtrialStateRate, /) -> Array:
    return jnp.stack(
        (
            rate.voltage_mV_per_ms,
            rate.m_per_ms,
            rate.h_per_ms,
            rate.j_per_ms,
            rate.oa_per_ms,
            rate.oi_per_ms,
            rate.ua_per_ms,
            rate.ui_per_ms,
            rate.xr_per_ms,
            rate.xs_per_ms,
            rate.d_per_ms,
            rate.f_per_ms,
            rate.f_ca_per_ms,
            rate.calcium_i_mM_per_ms,
            rate.calcium_sr_mM_per_ms,
        ),
        axis=-1,
    )


def _atrial_reaction_currents(currents: AtrialCurrents, /) -> Array:
    return jnp.stack(
        (
            currents.fast_sodium,
            currents.inward_rectifier_potassium,
            currents.transient_outward_potassium,
            currents.ultrarapid_potassium,
            currents.rapid_potassium,
            currents.slow_potassium,
            currents.l_type_calcium,
            currents.background_sodium,
            currents.background_calcium,
            currents.sodium_potassium_pump,
            currents.sodium_calcium_exchanger,
            currents.sarcolemmal_calcium_pump,
        ),
        axis=-1,
    )


@dataclass(frozen=True)
class CourtemancheAtrialReactionAdapter:
    """Final-axis reaction adapter for one homogeneous typed atrial model."""

    cell_model: CourtemancheAtrialModel = field(
        default_factory=lambda: CourtemancheAtrialParameters().prepare()
    )
    scaling: CardiacMembraneScaling = field(default_factory=CardiacMembraneScaling)
    model_id: str = field(init=False)
    default_parameters: Array = field(init=False, repr=False, compare=False)

    state_layout: ClassVar[CardiacReactionStateLayout] = CardiacReactionStateLayout(
        AtrialStateLayout().names,
        ("mV",) + ("1",) * 12 + ("mM", "mM"),
        AtrialStateLayout().names[1:13],
        AtrialStateLayout().names[13:],
    )
    parameter_layout: ClassVar[CardiacReactionParameterLayout] = (
        CardiacReactionParameterLayout(
            _ATRIAL_REACTION_PARAMETER_NAMES,
            (
                "mV",
                "C/mmol",
                "mM",
                "mM",
                "mM",
                "mM",
                "mM",
                "nS/pF",
                "nS/pF",
                "nS/pF",
                "1",
                "nS/pF",
                "nS/pF",
                "nS/pF",
                "nS/pF",
                "nS/pF",
                "pA/pF",
                "pA/pF",
                "pA/pF",
                "1",
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
        "I_K1",
        "I_to",
        "I_Kur",
        "I_Kr",
        "I_Ks",
        "I_CaL",
        "I_bNa",
        "I_bCa",
        "I_NaK",
        "I_NaCa",
        "I_CaP",
    )

    def __post_init__(self) -> None:
        if not isinstance(self.cell_model, CourtemancheAtrialModel):
            raise TypeError("cell_model must be CourtemancheAtrialModel.")
        if not isinstance(self.scaling, CardiacMembraneScaling):
            raise TypeError("scaling must be CardiacMembraneScaling.")
        object.__setattr__(
            self,
            "default_parameters",
            jnp.asarray(_atrial_parameter_values(self.cell_model.parameters)),
        )
        object.__setattr__(
            self,
            "model_id",
            canonical_fingerprint(
                {
                    "kind": "cardiovascular-atrial-reaction-adapter-v1",
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
        return _atrial_reaction_state(
            self.cell_model.initialize(batch_shape, dtype=dtype)
        )

    def admissible(self, state: Array, parameters: Array | None = None) -> Array:
        resolved = self.state_layout.require_shape(state)
        native = _atrial_native_state(resolved)
        configured = self._parameters(parameters, resolved.dtype)
        return self.cell_model.admissibility(
            native
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
        native = _atrial_native_state(resolved)
        stimulus = jnp.asarray(stimulus_current_uA_per_mm2, dtype=resolved.dtype)
        system = self.cell_model.rates(
            native,
            applied_current_pA_per_pF=(stimulus / self.membrane_capacitance_uF_per_mm2),
        )
        state_rate = _atrial_reaction_rate(system.state_rate)
        current_density = (
            _atrial_reaction_currents(system.currents)
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
                "atrial reaction state must have final axis size "
                f"{self.state_layout.state_count}, received {array.shape}."
            )
        if not np.all(np.isfinite(array)):
            raise ValueError("atrial reaction state must be finite.")
        parameter_array = None if parameters is None else jnp.asarray(parameters)
        if not np.all(np.asarray(self.admissible(jnp.asarray(array), parameter_array))):
            raise ValueError(
                "atrial reaction state or configured parameters are inadmissible."
            )


__all__ = [
    "AtrialAdmissibilityEvidence",
    "AtrialAdmissibilityStatus",
    "AtrialCalciumOutput",
    "AtrialCurrents",
    "AtrialPhenotype",
    "AtrialRateSystem",
    "AtrialState",
    "AtrialStateLayout",
    "AtrialStateRate",
    "CourtemancheAtrialModel",
    "CourtemancheAtrialParameters",
    "CourtemancheAtrialReactionAdapter",
]
