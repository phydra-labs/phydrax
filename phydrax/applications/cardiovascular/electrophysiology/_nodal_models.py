#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Distinct sinoatrial and atrioventricular nodal cell formulations.

The SAN subsystem is a qualified Zhang-2000 peripheral-node reduction with
``I_f``, L/T-type calcium and delayed rectifiers.  The AV subsystem is a
qualified Inada-2009 compact-node reduction with a separate sodium/``I_to``
schema.  They intentionally do not share a padded ventricular state or a
boolean phenotype branch.
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


class SinoatrialPhenotype(Enum):
    """Qualified autonomous sinoatrial identity."""

    RABBIT_PERIPHERAL_ZHANG2000_REDUCED = "rabbit-peripheral-san-zhang2000-reduced"


class AtrioventricularPhenotype(Enum):
    """Qualified compact atrioventricular-node identity."""

    RABBIT_COMPACT_INADA2009_REDUCED = "rabbit-compact-av-node-inada2009-reduced"


class NodalAdmissibilityStatus(IntFlag):
    """Fail-closed nodal state status bits."""

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


def _full(value: float, shape: tuple[int, ...], dtype: jnp.dtype, /) -> Array:
    return jnp.full(shape, value, dtype=dtype)


def _admissibility(
    packed: Array, gate_stop: int, calcium: Array, voltage: Array, /
) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
    gates = packed[1:gate_stop]
    finite = jnp.all(jnp.isfinite(packed), axis=0)
    gate_minimum = jnp.min(gates, axis=0)
    gate_maximum = jnp.max(gates, axis=0)
    gate_violation = jnp.maximum(jnp.maximum(-gate_minimum, gate_maximum - 1.0), 0.0)
    voltage_magnitude = jnp.abs(voltage)
    status = jnp.zeros_like(voltage, dtype=jnp.int32)
    status = jnp.where(finite, status, status | int(NodalAdmissibilityStatus.NONFINITE))
    status = jnp.where(
        gate_violation <= 1.0e-6,
        status,
        status | int(NodalAdmissibilityStatus.GATE_OUT_OF_RANGE),
    )
    status = jnp.where(
        calcium > 0.0, status, status | int(NodalAdmissibilityStatus.NONPOSITIVE_CALCIUM)
    )
    status = jnp.where(
        voltage_magnitude <= 200.0,
        status,
        status | int(NodalAdmissibilityStatus.VOLTAGE_OUT_OF_RANGE),
    )
    return (
        finite,
        gate_minimum,
        gate_maximum,
        gate_violation,
        calcium,
        voltage_magnitude,
        status,
        status == int(NodalAdmissibilityStatus.SUCCESS),
    )


class ZhangSinoatrialParameters(StrictModule, NonTrainableState):
    """Coefficient plan for the reduced Zhang peripheral SAN cell."""

    phenotype: SinoatrialPhenotype = eqx.field(static=True)
    rtf_mV: float = eqx.field(static=True)
    potassium_i_mM: float = eqx.field(static=True)
    potassium_o_mM: float = eqx.field(static=True)
    calcium_o_mM: float = eqx.field(static=True)
    funny_reversal_mV: float = eqx.field(static=True)
    background_reversal_mV: float = eqx.field(static=True)
    g_f: float = eqx.field(static=True)
    g_cal: float = eqx.field(static=True)
    g_cat: float = eqx.field(static=True)
    g_kr: float = eqx.field(static=True)
    g_ks: float = eqx.field(static=True)
    g_k1: float = eqx.field(static=True)
    g_background: float = eqx.field(static=True)
    ca_current_scale: float = eqx.field(static=True)
    uptake_max: float = eqx.field(static=True)
    uptake_half_mM: float = eqx.field(static=True)
    sr_leak_rate: float = eqx.field(static=True)
    sr_release_rate: float = eqx.field(static=True)
    sr_volume_ratio: float = eqx.field(static=True)
    parameter_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        rtf_mV: float = 26.7137607,
        potassium_i_mM: float = 140.0,
        potassium_o_mM: float = 5.4,
        calcium_o_mM: float = 2.0,
        funny_reversal_mV: float = -30.0,
        background_reversal_mV: float = -35.0,
        g_f: float = 0.012,
        g_cal: float = 0.090,
        g_cat: float = 0.040,
        g_kr: float = 0.030,
        g_ks: float = 0.004,
        g_k1: float = 0.005,
        g_background: float = 0.001,
        ca_current_scale: float = 2.5e-5,
        uptake_max: float = 0.0012,
        uptake_half_mM: float = 0.0005,
        sr_leak_rate: float = 1.0e-4,
        sr_release_rate: float = 0.08,
        sr_volume_ratio: float = 8.0,
    ):
        values = {
            "rtf_mV": _positive(rtf_mV, "rtf_mV"),
            "potassium_i_mM": _positive(potassium_i_mM, "potassium_i_mM"),
            "potassium_o_mM": _positive(potassium_o_mM, "potassium_o_mM"),
            "calcium_o_mM": _positive(calcium_o_mM, "calcium_o_mM"),
            "funny_reversal_mV": _finite(funny_reversal_mV, "funny_reversal_mV"),
            "background_reversal_mV": _finite(
                background_reversal_mV, "background_reversal_mV"
            ),
            "g_f": _positive(g_f, "g_f"),
            "g_cal": _positive(g_cal, "g_cal"),
            "g_cat": _positive(g_cat, "g_cat"),
            "g_kr": _positive(g_kr, "g_kr"),
            "g_ks": _positive(g_ks, "g_ks"),
            "g_k1": _nonnegative(g_k1, "g_k1"),
            "g_background": _positive(g_background, "g_background"),
            "ca_current_scale": _positive(ca_current_scale, "ca_current_scale"),
            "uptake_max": _positive(uptake_max, "uptake_max"),
            "uptake_half_mM": _positive(uptake_half_mM, "uptake_half_mM"),
            "sr_leak_rate": _positive(sr_leak_rate, "sr_leak_rate"),
            "sr_release_rate": _positive(sr_release_rate, "sr_release_rate"),
            "sr_volume_ratio": _positive(sr_volume_ratio, "sr_volume_ratio"),
        }
        self.phenotype = SinoatrialPhenotype.RABBIT_PERIPHERAL_ZHANG2000_REDUCED
        for name, value in values.items():
            object.__setattr__(self, name, value)
        self.parameter_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-san-zhang2000-reduced-parameters-v1",
                "phenotype": self.phenotype.value,
                "coefficients": values,
            }
        )

    def prepare(self) -> ZhangSinoatrialModel:
        return ZhangSinoatrialModel(self)


class SinoatrialState(StrictModule):
    """Fixed Zhang-reduced SAN structure-of-arrays state."""

    voltage_mV: Array
    y_f: Array
    d_l: Array
    f_l: Array
    d_t: Array
    f_t: Array
    x_r: Array
    x_s: Array
    calcium_i_mM: Array
    calcium_sr_mM: Array


class SinoatrialStateRate(StrictModule):
    voltage_mV_per_ms: Array
    y_f_per_ms: Array
    d_l_per_ms: Array
    f_l_per_ms: Array
    d_t_per_ms: Array
    f_t_per_ms: Array
    x_r_per_ms: Array
    x_s_per_ms: Array
    calcium_i_mM_per_ms: Array
    calcium_sr_mM_per_ms: Array


class SinoatrialStateLayout(StrictModule, NonTrainableState):
    names: tuple[str, ...] = eqx.field(static=True)
    state_size: int = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(self):
        names = (
            "voltage_mV",
            "y_f",
            "d_l",
            "f_l",
            "d_t",
            "f_t",
            "x_r",
            "x_s",
            "calcium_i_mM",
            "calcium_sr_mM",
        )
        self.names = names
        self.state_size = len(names)
        self.layout_id = canonical_fingerprint(
            {"kind": "cardiovascular-san-state-layout-v1", "names": list(names)}
        )

    def index(self, name: str, /) -> int:
        if name not in self.names:
            raise KeyError(f"Unknown sinoatrial state field {name!r}.")
        return self.names.index(name)

    def pack(self, state: SinoatrialState, /) -> Array:
        if not isinstance(state, SinoatrialState):
            raise TypeError("state must be SinoatrialState.")
        return jnp.stack(
            (
                state.voltage_mV,
                state.y_f,
                state.d_l,
                state.f_l,
                state.d_t,
                state.f_t,
                state.x_r,
                state.x_s,
                state.calcium_i_mM,
                state.calcium_sr_mM,
            ),
            axis=0,
        )

    def unpack(self, values: Array, /) -> SinoatrialState:
        array = jnp.asarray(values)
        if array.ndim == 0 or array.shape[0] != self.state_size:
            raise ValueError(
                f"Packed sinoatrial state must have leading size {self.state_size}."
            )
        return SinoatrialState(*(array[index] for index in range(self.state_size)))


class SinoatrialCurrents(StrictModule):
    """Outward-positive SAN current densities in pA/pF."""

    funny: Array
    l_type_calcium: Array
    t_type_calcium: Array
    rapid_potassium: Array
    slow_potassium: Array
    inward_rectifier_potassium: Array
    background: Array
    total_ionic: Array


class SinoatrialCalciumOutput(StrictModule):
    membrane_current_pA_per_pF: Array
    membrane_flux_mM_per_ms: Array
    uptake_flux_mM_per_ms: Array
    leak_flux_mM_per_ms: Array
    release_flux_mM_per_ms: Array
    net_cytosolic_flux_mM_per_ms: Array


class SinoatrialAdmissibilityEvidence(StrictModule):
    finite: Array
    gate_minimum: Array
    gate_maximum: Array
    maximum_gate_violation: Array
    minimum_calcium_mM: Array
    maximum_voltage_magnitude_mV: Array
    status: Array
    successful: Array


class SinoatrialRateSystem(StrictModule):
    state_rate: SinoatrialStateRate
    currents: SinoatrialCurrents
    calcium: SinoatrialCalciumOutput
    evidence: SinoatrialAdmissibilityEvidence
    gate_steady_state: Array
    gate_time_constant_ms: Array


class ZhangSinoatrialModel(StrictModule, NonTrainableState):
    """Prepared autonomous Zhang-2000 reduced peripheral SAN model."""

    parameters: ZhangSinoatrialParameters
    layout: SinoatrialStateLayout
    model_id: str = eqx.field(static=True)

    def __init__(self, parameters: ZhangSinoatrialParameters, /):
        if not isinstance(parameters, ZhangSinoatrialParameters):
            raise TypeError("parameters must be ZhangSinoatrialParameters.")
        layout = SinoatrialStateLayout()
        self.parameters = parameters
        self.layout = layout
        self.model_id = canonical_fingerprint(
            {
                "kind": "prepared-cardiovascular-san-zhang2000-reduced-v1",
                "parameters": parameters.parameter_id,
                "layout": layout.layout_id,
            }
        )

    def initialize(
        self, batch_shape: Sequence[int] = (), *, dtype: jnp.dtype | None = None
    ) -> SinoatrialState:
        shape = _shape(batch_shape)
        resolved_dtype = jnp.asarray(0.0).dtype if dtype is None else jnp.dtype(dtype)
        values = (-58.0, 0.20, 0.004, 0.95, 0.020, 0.80, 0.10, 0.03, 1.0e-4, 0.80)
        return SinoatrialState(*(_full(value, shape, resolved_dtype) for value in values))

    def currents(self, state: SinoatrialState, /) -> SinoatrialCurrents:
        if not isinstance(state, SinoatrialState):
            raise TypeError("state must be SinoatrialState.")
        p = self.parameters
        v = state.voltage_mV
        ek = p.rtf_mV * jnp.log(p.potassium_o_mM / p.potassium_i_mM)
        eca = 0.5 * p.rtf_mV * jnp.log(p.calcium_o_mM / state.calcium_i_mM)
        i_f = p.g_f * state.y_f * (v - p.funny_reversal_mV)
        i_cal = p.g_cal * state.d_l * state.f_l * (v - eca)
        i_cat = p.g_cat * state.d_t * state.f_t * (v - eca)
        i_kr = p.g_kr * state.x_r * (v - ek) / (1.0 + _exp((v + 15.0) / 22.4))
        i_ks = p.g_ks * state.x_s**2 * (v - ek)
        i_k1 = p.g_k1 * (v - ek) / (1.0 + _exp(0.07 * (v + 80.0)))
        i_background = p.g_background * (v - p.background_reversal_mV)
        total = i_f + i_cal + i_cat + i_kr + i_ks + i_k1 + i_background
        return SinoatrialCurrents(
            i_f, i_cal, i_cat, i_kr, i_ks, i_k1, i_background, total
        )

    def calcium_output(
        self, state: SinoatrialState, currents: SinoatrialCurrents, /
    ) -> SinoatrialCalciumOutput:
        if not isinstance(currents, SinoatrialCurrents):
            raise TypeError("currents must be SinoatrialCurrents.")
        p = self.parameters
        membrane_current = currents.l_type_calcium + currents.t_type_calcium
        membrane_flux = -p.ca_current_scale * membrane_current
        uptake = (
            p.uptake_max * state.calcium_i_mM / (p.uptake_half_mM + state.calcium_i_mM)
        )
        leak = p.sr_leak_rate * (state.calcium_sr_mM - state.calcium_i_mM)
        release = (
            p.sr_release_rate
            * state.d_l
            * state.f_l
            * (state.calcium_sr_mM - state.calcium_i_mM)
        )
        net = membrane_flux + leak + release - uptake
        return SinoatrialCalciumOutput(
            membrane_current, membrane_flux, uptake, leak, release, net
        )

    def admissibility(self, state: SinoatrialState, /) -> SinoatrialAdmissibilityEvidence:
        packed = self.layout.pack(state)
        calcium = jnp.minimum(state.calcium_i_mM, state.calcium_sr_mM)
        return SinoatrialAdmissibilityEvidence(
            *_admissibility(packed, 8, calcium, state.voltage_mV)
        )

    def rates(
        self,
        state: SinoatrialState,
        /,
        *,
        applied_current_pA_per_pF: Array | float = 0.0,
    ) -> SinoatrialRateSystem:
        currents = self.currents(state)
        calcium = self.calcium_output(state, currents)
        evidence = self.admissibility(state)
        v = state.voltage_mV
        y_inf = jax.nn.sigmoid(-(v + 80.0) / 10.0)
        tau_y = 800.0 / (_exp(-(v + 71.5) / 20.27) + _exp((v + 71.5) / 20.27)) + 20.0
        dl_inf = jax.nn.sigmoid((v + 22.3) / 6.0)
        tau_dl = 0.6 + 1.0 / (_exp(-0.05 * (v + 6.0)) + _exp(0.09 * (v + 14.0)))
        fl_inf = jax.nn.sigmoid(-(v + 45.0) / 5.0)
        tau_fl = 20.0 + 110.0 * jax.nn.sigmoid(-(v + 35.0) / 5.0)
        dt_inf = jax.nn.sigmoid((v + 35.0) / 6.0)
        tau_dt = 1.0 + 1.0 / (_exp((v + 35.0) / 30.0) + _exp(-(v + 35.0) / 30.0))
        ft_inf = jax.nn.sigmoid(-(v + 65.0) / 6.0)
        tau_ft = 20.0 + 50.0 * jax.nn.sigmoid(-(v + 40.0) / 5.0)
        xr_inf = jax.nn.sigmoid((v + 20.0) / 8.0)
        tau_xr = 20.0 + 100.0 / (_exp((v + 20.0) / 20.0) + _exp(-(v + 20.0) / 20.0))
        xs_inf = jax.nn.sigmoid((v - 5.0) / 14.0)
        tau_xs = 100.0 + 400.0 / (_exp((v + 20.0) / 20.0) + _exp(-(v + 20.0) / 20.0))
        applied = jnp.asarray(applied_current_pA_per_pF, dtype=v.dtype)
        rate = SinoatrialStateRate(
            -(currents.total_ionic + applied),
            (y_inf - state.y_f) / tau_y,
            (dl_inf - state.d_l) / tau_dl,
            (fl_inf - state.f_l) / tau_fl,
            (dt_inf - state.d_t) / tau_dt,
            (ft_inf - state.f_t) / tau_ft,
            (xr_inf - state.x_r) / tau_xr,
            (xs_inf - state.x_s) / tau_xs,
            calcium.net_cytosolic_flux_mM_per_ms,
            self.parameters.sr_volume_ratio
            * (
                calcium.uptake_flux_mM_per_ms
                - calcium.leak_flux_mM_per_ms
                - calcium.release_flux_mM_per_ms
            ),
        )
        gate_steady_state = jnp.stack(
            (y_inf, dl_inf, fl_inf, dt_inf, ft_inf, xr_inf, xs_inf),
            axis=-1,
        )
        gate_time_constant_ms = jnp.stack(
            (tau_y, tau_dl, tau_fl, tau_dt, tau_ft, tau_xr, tau_xs),
            axis=-1,
        )
        return SinoatrialRateSystem(
            rate,
            currents,
            calcium,
            evidence,
            gate_steady_state,
            gate_time_constant_ms,
        )


class InadaAtrioventricularParameters(StrictModule, NonTrainableState):
    """Coefficient plan for the reduced Inada compact AV-node cell."""

    phenotype: AtrioventricularPhenotype = eqx.field(static=True)
    rtf_mV: float = eqx.field(static=True)
    sodium_i_mM: float = eqx.field(static=True)
    sodium_o_mM: float = eqx.field(static=True)
    potassium_i_mM: float = eqx.field(static=True)
    potassium_o_mM: float = eqx.field(static=True)
    calcium_o_mM: float = eqx.field(static=True)
    funny_reversal_mV: float = eqx.field(static=True)
    background_reversal_mV: float = eqx.field(static=True)
    g_na: float = eqx.field(static=True)
    g_cal: float = eqx.field(static=True)
    g_to: float = eqx.field(static=True)
    g_kr: float = eqx.field(static=True)
    g_k1: float = eqx.field(static=True)
    g_f: float = eqx.field(static=True)
    g_background: float = eqx.field(static=True)
    ca_current_scale: float = eqx.field(static=True)
    ca_removal_rate: float = eqx.field(static=True)
    resting_calcium_mM: float = eqx.field(static=True)
    parameter_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        rtf_mV: float = 26.7137607,
        sodium_i_mM: float = 8.0,
        sodium_o_mM: float = 140.0,
        potassium_i_mM: float = 140.0,
        potassium_o_mM: float = 5.4,
        calcium_o_mM: float = 2.0,
        funny_reversal_mV: float = -30.0,
        background_reversal_mV: float = -40.0,
        g_na: float = 0.30,
        g_cal: float = 0.050,
        g_to: float = 0.020,
        g_kr: float = 0.035,
        g_k1: float = 0.003,
        g_f: float = 0.004,
        g_background: float = 0.001,
        ca_current_scale: float = 2.0e-5,
        ca_removal_rate: float = 0.020,
        resting_calcium_mM: float = 1.0e-4,
    ):
        values = {
            "rtf_mV": _positive(rtf_mV, "rtf_mV"),
            "sodium_i_mM": _positive(sodium_i_mM, "sodium_i_mM"),
            "sodium_o_mM": _positive(sodium_o_mM, "sodium_o_mM"),
            "potassium_i_mM": _positive(potassium_i_mM, "potassium_i_mM"),
            "potassium_o_mM": _positive(potassium_o_mM, "potassium_o_mM"),
            "calcium_o_mM": _positive(calcium_o_mM, "calcium_o_mM"),
            "funny_reversal_mV": _finite(funny_reversal_mV, "funny_reversal_mV"),
            "background_reversal_mV": _finite(
                background_reversal_mV, "background_reversal_mV"
            ),
            "g_na": _positive(g_na, "g_na"),
            "g_cal": _positive(g_cal, "g_cal"),
            "g_to": _positive(g_to, "g_to"),
            "g_kr": _positive(g_kr, "g_kr"),
            "g_k1": _nonnegative(g_k1, "g_k1"),
            "g_f": _positive(g_f, "g_f"),
            "g_background": _positive(g_background, "g_background"),
            "ca_current_scale": _positive(ca_current_scale, "ca_current_scale"),
            "ca_removal_rate": _positive(ca_removal_rate, "ca_removal_rate"),
            "resting_calcium_mM": _positive(resting_calcium_mM, "resting_calcium_mM"),
        }
        self.phenotype = AtrioventricularPhenotype.RABBIT_COMPACT_INADA2009_REDUCED
        for name, value in values.items():
            object.__setattr__(self, name, value)
        self.parameter_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-av-node-inada2009-reduced-parameters-v1",
                "phenotype": self.phenotype.value,
                "coefficients": values,
            }
        )

    def prepare(self) -> InadaAtrioventricularModel:
        return InadaAtrioventricularModel(self)


class AtrioventricularState(StrictModule):
    """Fixed Inada-reduced AV-node SoA; distinct from the SAN layout."""

    voltage_mV: Array
    m: Array
    h: Array
    d_l: Array
    f_l: Array
    r_to: Array
    q_to: Array
    x_r: Array
    y_f: Array
    calcium_i_mM: Array


class AtrioventricularStateRate(StrictModule):
    voltage_mV_per_ms: Array
    m_per_ms: Array
    h_per_ms: Array
    d_l_per_ms: Array
    f_l_per_ms: Array
    r_to_per_ms: Array
    q_to_per_ms: Array
    x_r_per_ms: Array
    y_f_per_ms: Array
    calcium_i_mM_per_ms: Array


class AtrioventricularStateLayout(StrictModule, NonTrainableState):
    names: tuple[str, ...] = eqx.field(static=True)
    state_size: int = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(self):
        names = (
            "voltage_mV",
            "m",
            "h",
            "d_l",
            "f_l",
            "r_to",
            "q_to",
            "x_r",
            "y_f",
            "calcium_i_mM",
        )
        self.names = names
        self.state_size = len(names)
        self.layout_id = canonical_fingerprint(
            {"kind": "cardiovascular-av-node-state-layout-v1", "names": list(names)}
        )

    def index(self, name: str, /) -> int:
        if name not in self.names:
            raise KeyError(f"Unknown atrioventricular state field {name!r}.")
        return self.names.index(name)

    def pack(self, state: AtrioventricularState, /) -> Array:
        if not isinstance(state, AtrioventricularState):
            raise TypeError("state must be AtrioventricularState.")
        return jnp.stack(
            (
                state.voltage_mV,
                state.m,
                state.h,
                state.d_l,
                state.f_l,
                state.r_to,
                state.q_to,
                state.x_r,
                state.y_f,
                state.calcium_i_mM,
            ),
            axis=0,
        )

    def unpack(self, values: Array, /) -> AtrioventricularState:
        array = jnp.asarray(values)
        if array.ndim == 0 or array.shape[0] != self.state_size:
            raise ValueError(
                f"Packed atrioventricular state must have leading size {self.state_size}."
            )
        return AtrioventricularState(*(array[index] for index in range(self.state_size)))


class AtrioventricularCurrents(StrictModule):
    """Outward-positive compact AV-node current densities in pA/pF."""

    fast_sodium: Array
    l_type_calcium: Array
    transient_outward_potassium: Array
    rapid_potassium: Array
    inward_rectifier_potassium: Array
    funny: Array
    background: Array
    total_ionic: Array


class AtrioventricularCalciumOutput(StrictModule):
    membrane_current_pA_per_pF: Array
    membrane_flux_mM_per_ms: Array
    removal_flux_mM_per_ms: Array
    net_cytosolic_flux_mM_per_ms: Array


class AtrioventricularAdmissibilityEvidence(StrictModule):
    finite: Array
    gate_minimum: Array
    gate_maximum: Array
    maximum_gate_violation: Array
    minimum_calcium_mM: Array
    maximum_voltage_magnitude_mV: Array
    status: Array
    successful: Array


class AtrioventricularRateSystem(StrictModule):
    state_rate: AtrioventricularStateRate
    currents: AtrioventricularCurrents
    calcium: AtrioventricularCalciumOutput
    evidence: AtrioventricularAdmissibilityEvidence
    gate_steady_state: Array
    gate_time_constant_ms: Array


class InadaAtrioventricularModel(StrictModule, NonTrainableState):
    """Prepared autonomous Inada-2009 reduced compact AV-node model."""

    parameters: InadaAtrioventricularParameters
    layout: AtrioventricularStateLayout
    model_id: str = eqx.field(static=True)

    def __init__(self, parameters: InadaAtrioventricularParameters, /):
        if not isinstance(parameters, InadaAtrioventricularParameters):
            raise TypeError("parameters must be InadaAtrioventricularParameters.")
        layout = AtrioventricularStateLayout()
        self.parameters = parameters
        self.layout = layout
        self.model_id = canonical_fingerprint(
            {
                "kind": "prepared-cardiovascular-av-node-inada2009-reduced-v1",
                "parameters": parameters.parameter_id,
                "layout": layout.layout_id,
            }
        )

    def initialize(
        self, batch_shape: Sequence[int] = (), *, dtype: jnp.dtype | None = None
    ) -> AtrioventricularState:
        shape = _shape(batch_shape)
        resolved_dtype = jnp.asarray(0.0).dtype if dtype is None else jnp.dtype(dtype)
        values = (-60.0, 0.05, 0.80, 0.003, 0.90, 0.010, 0.90, 0.05, 0.12, 1.0e-4)
        return AtrioventricularState(
            *(_full(value, shape, resolved_dtype) for value in values)
        )

    def currents(self, state: AtrioventricularState, /) -> AtrioventricularCurrents:
        if not isinstance(state, AtrioventricularState):
            raise TypeError("state must be AtrioventricularState.")
        p = self.parameters
        v = state.voltage_mV
        ena = p.rtf_mV * jnp.log(p.sodium_o_mM / p.sodium_i_mM)
        ek = p.rtf_mV * jnp.log(p.potassium_o_mM / p.potassium_i_mM)
        eca = 0.5 * p.rtf_mV * jnp.log(p.calcium_o_mM / state.calcium_i_mM)
        i_na = p.g_na * state.m**3 * state.h * (v - ena)
        i_cal = p.g_cal * state.d_l * state.f_l * (v - eca)
        i_to = p.g_to * state.r_to * state.q_to * (v - ek)
        i_kr = p.g_kr * state.x_r * (v - ek) / (1.0 + _exp((v + 15.0) / 22.4))
        i_k1 = p.g_k1 * (v - ek) / (1.0 + _exp(0.07 * (v + 80.0)))
        i_f = p.g_f * state.y_f * (v - p.funny_reversal_mV)
        i_background = p.g_background * (v - p.background_reversal_mV)
        total = i_na + i_cal + i_to + i_kr + i_k1 + i_f + i_background
        return AtrioventricularCurrents(
            i_na, i_cal, i_to, i_kr, i_k1, i_f, i_background, total
        )

    def calcium_output(
        self, state: AtrioventricularState, currents: AtrioventricularCurrents, /
    ) -> AtrioventricularCalciumOutput:
        if not isinstance(currents, AtrioventricularCurrents):
            raise TypeError("currents must be AtrioventricularCurrents.")
        membrane_flux = -self.parameters.ca_current_scale * currents.l_type_calcium
        removal = self.parameters.ca_removal_rate * (
            state.calcium_i_mM - self.parameters.resting_calcium_mM
        )
        return AtrioventricularCalciumOutput(
            currents.l_type_calcium, membrane_flux, removal, membrane_flux - removal
        )

    def admissibility(
        self, state: AtrioventricularState, /
    ) -> AtrioventricularAdmissibilityEvidence:
        packed = self.layout.pack(state)
        return AtrioventricularAdmissibilityEvidence(
            *_admissibility(packed, 9, state.calcium_i_mM, state.voltage_mV)
        )

    def rates(
        self,
        state: AtrioventricularState,
        /,
        *,
        applied_current_pA_per_pF: Array | float = 0.0,
    ) -> AtrioventricularRateSystem:
        currents = self.currents(state)
        calcium = self.calcium_output(state, currents)
        evidence = self.admissibility(state)
        v = state.voltage_mV
        m_inf = jax.nn.sigmoid((v + 35.0) / 6.0)
        tau_m = 0.15 + 0.40 / (_exp((v + 35.0) / 20.0) + _exp(-(v + 35.0) / 20.0))
        h_inf = jax.nn.sigmoid(-(v + 58.0) / 6.0)
        tau_h = 2.0 + 12.0 * jax.nn.sigmoid(-(v + 40.0) / 5.0)
        dl_inf = jax.nn.sigmoid((v + 18.0) / 6.5)
        tau_dl = 0.8 + 1.2 / (_exp((v + 10.0) / 25.0) + _exp(-(v + 10.0) / 25.0))
        fl_inf = jax.nn.sigmoid(-(v + 35.0) / 7.0)
        tau_fl = 18.0 + 60.0 * jax.nn.sigmoid(-(v + 25.0) / 5.0)
        r_inf = jax.nn.sigmoid((v + 15.0) / 8.0)
        tau_r = 3.0 + 5.0 / (_exp((v + 15.0) / 30.0) + _exp(-(v + 15.0) / 30.0))
        q_inf = jax.nn.sigmoid(-(v + 45.0) / 5.0)
        tau_q = 20.0 + 35.0 * jax.nn.sigmoid(-(v + 30.0) / 5.0)
        xr_inf = jax.nn.sigmoid((v + 20.0) / 8.0)
        tau_xr = 25.0 + 80.0 / (_exp((v + 20.0) / 25.0) + _exp(-(v + 20.0) / 25.0))
        yf_inf = jax.nn.sigmoid(-(v + 78.0) / 9.0)
        tau_yf = 250.0 + 600.0 / (_exp((v + 70.0) / 25.0) + _exp(-(v + 70.0) / 25.0))
        applied = jnp.asarray(applied_current_pA_per_pF, dtype=v.dtype)
        rate = AtrioventricularStateRate(
            -(currents.total_ionic + applied),
            (m_inf - state.m) / tau_m,
            (h_inf - state.h) / tau_h,
            (dl_inf - state.d_l) / tau_dl,
            (fl_inf - state.f_l) / tau_fl,
            (r_inf - state.r_to) / tau_r,
            (q_inf - state.q_to) / tau_q,
            (xr_inf - state.x_r) / tau_xr,
            (yf_inf - state.y_f) / tau_yf,
            calcium.net_cytosolic_flux_mM_per_ms,
        )
        gate_steady_state = jnp.stack(
            (m_inf, h_inf, dl_inf, fl_inf, r_inf, q_inf, xr_inf, yf_inf),
            axis=-1,
        )
        gate_time_constant_ms = jnp.stack(
            (tau_m, tau_h, tau_dl, tau_fl, tau_r, tau_q, tau_xr, tau_yf),
            axis=-1,
        )
        return AtrioventricularRateSystem(
            rate,
            currents,
            calcium,
            evidence,
            gate_steady_state,
            gate_time_constant_ms,
        )


_SAN_REACTION_PARAMETER_NAMES = (
    "rtf_mV",
    "potassium_i_mM",
    "potassium_o_mM",
    "calcium_o_mM",
    "funny_reversal_mV",
    "background_reversal_mV",
    "g_f",
    "g_cal",
    "g_cat",
    "g_kr",
    "g_ks",
    "g_k1",
    "g_background",
    "ca_current_scale",
    "uptake_max",
    "uptake_half_mM",
    "sr_leak_rate",
    "sr_release_rate",
    "sr_volume_ratio",
)


def _san_parameter_values(parameters: ZhangSinoatrialParameters, /) -> tuple[float, ...]:
    return (
        parameters.rtf_mV,
        parameters.potassium_i_mM,
        parameters.potassium_o_mM,
        parameters.calcium_o_mM,
        parameters.funny_reversal_mV,
        parameters.background_reversal_mV,
        parameters.g_f,
        parameters.g_cal,
        parameters.g_cat,
        parameters.g_kr,
        parameters.g_ks,
        parameters.g_k1,
        parameters.g_background,
        parameters.ca_current_scale,
        parameters.uptake_max,
        parameters.uptake_half_mM,
        parameters.sr_leak_rate,
        parameters.sr_release_rate,
        parameters.sr_volume_ratio,
    )


def _san_reaction_state(state: SinoatrialState, /) -> Array:
    return jnp.stack(
        (
            state.voltage_mV,
            state.y_f,
            state.d_l,
            state.f_l,
            state.d_t,
            state.f_t,
            state.x_r,
            state.x_s,
            state.calcium_i_mM,
            state.calcium_sr_mM,
        ),
        axis=-1,
    )


def _san_native_state(state: Array, /) -> SinoatrialState:
    return SinoatrialState(*(state[..., index] for index in range(10)))


def _san_reaction_rate(rate: SinoatrialStateRate, /) -> Array:
    return jnp.stack(
        (
            rate.voltage_mV_per_ms,
            rate.y_f_per_ms,
            rate.d_l_per_ms,
            rate.f_l_per_ms,
            rate.d_t_per_ms,
            rate.f_t_per_ms,
            rate.x_r_per_ms,
            rate.x_s_per_ms,
            rate.calcium_i_mM_per_ms,
            rate.calcium_sr_mM_per_ms,
        ),
        axis=-1,
    )


def _san_reaction_currents(currents: SinoatrialCurrents, /) -> Array:
    return jnp.stack(
        (
            currents.funny,
            currents.l_type_calcium,
            currents.t_type_calcium,
            currents.rapid_potassium,
            currents.slow_potassium,
            currents.inward_rectifier_potassium,
            currents.background,
        ),
        axis=-1,
    )


@dataclass(frozen=True)
class ZhangSinoatrialReactionAdapter:
    """Final-axis reaction adapter for one homogeneous typed SAN model."""

    cell_model: ZhangSinoatrialModel = field(
        default_factory=lambda: ZhangSinoatrialParameters().prepare()
    )
    scaling: CardiacMembraneScaling = field(default_factory=CardiacMembraneScaling)
    model_id: str = field(init=False)
    default_parameters: Array = field(init=False, repr=False, compare=False)

    state_layout: ClassVar[CardiacReactionStateLayout] = CardiacReactionStateLayout(
        SinoatrialStateLayout().names,
        ("mV",) + ("1",) * 7 + ("mM", "mM"),
        SinoatrialStateLayout().names[1:8],
        SinoatrialStateLayout().names[8:],
    )
    parameter_layout: ClassVar[CardiacReactionParameterLayout] = (
        CardiacReactionParameterLayout(
            _SAN_REACTION_PARAMETER_NAMES,
            (
                "mV",
                "mM",
                "mM",
                "mM",
                "mV",
                "mV",
                "nS/pF",
                "nS/pF",
                "nS/pF",
                "nS/pF",
                "nS/pF",
                "nS/pF",
                "nS/pF",
                "mM ms^-1/(pA pF^-1)",
                "mM/ms",
                "mM",
                "1/ms",
                "1/ms",
                "1",
            ),
        )
    )
    current_names: ClassVar[tuple[str, ...]] = (
        "I_f",
        "I_CaL",
        "I_CaT",
        "I_Kr",
        "I_Ks",
        "I_K1",
        "I_background",
    )

    def __post_init__(self) -> None:
        if not isinstance(self.cell_model, ZhangSinoatrialModel):
            raise TypeError("cell_model must be ZhangSinoatrialModel.")
        if not isinstance(self.scaling, CardiacMembraneScaling):
            raise TypeError("scaling must be CardiacMembraneScaling.")
        object.__setattr__(
            self,
            "default_parameters",
            jnp.asarray(_san_parameter_values(self.cell_model.parameters)),
        )
        object.__setattr__(
            self,
            "model_id",
            canonical_fingerprint(
                {
                    "kind": "cardiovascular-san-reaction-adapter-v1",
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
        return _san_reaction_state(self.cell_model.initialize(batch_shape, dtype=dtype))

    def admissible(self, state: Array, parameters: Array | None = None) -> Array:
        resolved = self.state_layout.require_shape(state)
        configured = self._parameters(parameters, resolved.dtype)
        return self.cell_model.admissibility(
            _san_native_state(resolved)
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
        native = _san_native_state(resolved)
        stimulus = jnp.asarray(stimulus_current_uA_per_mm2, dtype=resolved.dtype)
        system = self.cell_model.rates(
            native,
            applied_current_pA_per_pF=(stimulus / self.membrane_capacitance_uF_per_mm2),
        )
        state_rate = _san_reaction_rate(system.state_rate)
        current_density = (
            _san_reaction_currents(system.currents) * self.membrane_capacitance_uF_per_mm2
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
                "SAN reaction state must have final axis size "
                f"{self.state_layout.state_count}, received {array.shape}."
            )
        parameter_array = None if parameters is None else jnp.asarray(parameters)
        if not np.all(np.isfinite(array)) or not np.all(
            np.asarray(self.admissible(jnp.asarray(array), parameter_array))
        ):
            raise ValueError(
                "SAN reaction state or configured parameters are inadmissible."
            )


_AV_REACTION_PARAMETER_NAMES = (
    "rtf_mV",
    "sodium_i_mM",
    "sodium_o_mM",
    "potassium_i_mM",
    "potassium_o_mM",
    "calcium_o_mM",
    "funny_reversal_mV",
    "background_reversal_mV",
    "g_na",
    "g_cal",
    "g_to",
    "g_kr",
    "g_k1",
    "g_f",
    "g_background",
    "ca_current_scale",
    "ca_removal_rate",
    "resting_calcium_mM",
)


def _av_parameter_values(
    parameters: InadaAtrioventricularParameters, /
) -> tuple[float, ...]:
    return (
        parameters.rtf_mV,
        parameters.sodium_i_mM,
        parameters.sodium_o_mM,
        parameters.potassium_i_mM,
        parameters.potassium_o_mM,
        parameters.calcium_o_mM,
        parameters.funny_reversal_mV,
        parameters.background_reversal_mV,
        parameters.g_na,
        parameters.g_cal,
        parameters.g_to,
        parameters.g_kr,
        parameters.g_k1,
        parameters.g_f,
        parameters.g_background,
        parameters.ca_current_scale,
        parameters.ca_removal_rate,
        parameters.resting_calcium_mM,
    )


def _av_reaction_state(state: AtrioventricularState, /) -> Array:
    return jnp.stack(
        (
            state.voltage_mV,
            state.m,
            state.h,
            state.d_l,
            state.f_l,
            state.r_to,
            state.q_to,
            state.x_r,
            state.y_f,
            state.calcium_i_mM,
        ),
        axis=-1,
    )


def _av_native_state(state: Array, /) -> AtrioventricularState:
    return AtrioventricularState(*(state[..., index] for index in range(10)))


def _av_reaction_rate(rate: AtrioventricularStateRate, /) -> Array:
    return jnp.stack(
        (
            rate.voltage_mV_per_ms,
            rate.m_per_ms,
            rate.h_per_ms,
            rate.d_l_per_ms,
            rate.f_l_per_ms,
            rate.r_to_per_ms,
            rate.q_to_per_ms,
            rate.x_r_per_ms,
            rate.y_f_per_ms,
            rate.calcium_i_mM_per_ms,
        ),
        axis=-1,
    )


def _av_reaction_currents(currents: AtrioventricularCurrents, /) -> Array:
    return jnp.stack(
        (
            currents.fast_sodium,
            currents.l_type_calcium,
            currents.transient_outward_potassium,
            currents.rapid_potassium,
            currents.inward_rectifier_potassium,
            currents.funny,
            currents.background,
        ),
        axis=-1,
    )


@dataclass(frozen=True)
class InadaAtrioventricularReactionAdapter:
    """Final-axis reaction adapter for one homogeneous typed AV-node model."""

    cell_model: InadaAtrioventricularModel = field(
        default_factory=lambda: InadaAtrioventricularParameters().prepare()
    )
    scaling: CardiacMembraneScaling = field(default_factory=CardiacMembraneScaling)
    model_id: str = field(init=False)
    default_parameters: Array = field(init=False, repr=False, compare=False)

    state_layout: ClassVar[CardiacReactionStateLayout] = CardiacReactionStateLayout(
        AtrioventricularStateLayout().names,
        ("mV",) + ("1",) * 8 + ("mM",),
        AtrioventricularStateLayout().names[1:9],
        AtrioventricularStateLayout().names[9:],
    )
    parameter_layout: ClassVar[CardiacReactionParameterLayout] = (
        CardiacReactionParameterLayout(
            _AV_REACTION_PARAMETER_NAMES,
            (
                "mV",
                "mM",
                "mM",
                "mM",
                "mM",
                "mM",
                "mV",
                "mV",
                "nS/pF",
                "nS/pF",
                "nS/pF",
                "nS/pF",
                "nS/pF",
                "nS/pF",
                "nS/pF",
                "mM ms^-1/(pA pF^-1)",
                "1/ms",
                "mM",
            ),
        )
    )
    current_names: ClassVar[tuple[str, ...]] = (
        "I_Na",
        "I_CaL",
        "I_to",
        "I_Kr",
        "I_K1",
        "I_f",
        "I_background",
    )

    def __post_init__(self) -> None:
        if not isinstance(self.cell_model, InadaAtrioventricularModel):
            raise TypeError("cell_model must be InadaAtrioventricularModel.")
        if not isinstance(self.scaling, CardiacMembraneScaling):
            raise TypeError("scaling must be CardiacMembraneScaling.")
        object.__setattr__(
            self,
            "default_parameters",
            jnp.asarray(_av_parameter_values(self.cell_model.parameters)),
        )
        object.__setattr__(
            self,
            "model_id",
            canonical_fingerprint(
                {
                    "kind": "cardiovascular-av-node-reaction-adapter-v1",
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
        return _av_reaction_state(self.cell_model.initialize(batch_shape, dtype=dtype))

    def admissible(self, state: Array, parameters: Array | None = None) -> Array:
        resolved = self.state_layout.require_shape(state)
        configured = self._parameters(parameters, resolved.dtype)
        return self.cell_model.admissibility(
            _av_native_state(resolved)
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
        native = _av_native_state(resolved)
        stimulus = jnp.asarray(stimulus_current_uA_per_mm2, dtype=resolved.dtype)
        system = self.cell_model.rates(
            native,
            applied_current_pA_per_pF=(stimulus / self.membrane_capacitance_uF_per_mm2),
        )
        state_rate = _av_reaction_rate(system.state_rate)
        current_density = (
            _av_reaction_currents(system.currents) * self.membrane_capacitance_uF_per_mm2
        )
        total_current = jnp.sum(current_density, axis=-1)
        valid = system.evidence.successful & self._parameter_admissible(configured)
        nan = jnp.asarray(jnp.nan, dtype=resolved.dtype)
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
            calcium_sr_flux_mM_per_ms=jnp.where(
                valid, jnp.zeros_like(native.calcium_i_mM), nan
            ),
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
                "AV-node reaction state must have final axis size "
                f"{self.state_layout.state_count}, received {array.shape}."
            )
        parameter_array = None if parameters is None else jnp.asarray(parameters)
        if not np.all(np.isfinite(array)) or not np.all(
            np.asarray(self.admissible(jnp.asarray(array), parameter_array))
        ):
            raise ValueError(
                "AV-node reaction state or configured parameters are inadmissible."
            )


__all__ = [
    "AtrioventricularAdmissibilityEvidence",
    "AtrioventricularCalciumOutput",
    "AtrioventricularCurrents",
    "AtrioventricularPhenotype",
    "AtrioventricularRateSystem",
    "AtrioventricularState",
    "AtrioventricularStateLayout",
    "AtrioventricularStateRate",
    "InadaAtrioventricularModel",
    "InadaAtrioventricularParameters",
    "InadaAtrioventricularReactionAdapter",
    "NodalAdmissibilityStatus",
    "SinoatrialAdmissibilityEvidence",
    "SinoatrialCalciumOutput",
    "SinoatrialCurrents",
    "SinoatrialPhenotype",
    "SinoatrialRateSystem",
    "SinoatrialState",
    "SinoatrialStateLayout",
    "SinoatrialStateRate",
    "ZhangSinoatrialModel",
    "ZhangSinoatrialParameters",
    "ZhangSinoatrialReactionAdapter",
]
