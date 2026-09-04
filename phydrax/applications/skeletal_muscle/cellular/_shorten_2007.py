#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Shorten--O'Callaghan--Davidson--Soboleva fast-twitch muscle cell.

The equations are an independent transcription of the CellML mathematics at
PMR changeset ``637da9ef28f7992e40fe79947364a51a38ec818c``.  The source is
CC BY 3.0 and its exact 184890-byte payload has SHA-256
``e14e2aeffeb7b935017414a5ef53c06e43ed6b5fd4d7a92f07e0518b48b413c1``.
The implementation is not generated CellML code and has no CellML runtime.

Current is positive outward except ``stimulus_current_uA_per_cm2``, which is
positive inward as in source algebraic ``wal_environment/I_HH``.  Time is ms,
voltage mV, membrane current density uA/cm2, ionic concentrations mM, and the
calcium/crossbridge subsystem uses uM where named.  The source's force-bearing
observable is the post-power-stroke attached crossbridge concentration ``A_2``;
this model owns that tension driver and never combines it with D1, De Groote,
or continuum force laws.
"""

from __future__ import annotations

from enum import IntFlag
from math import isfinite
from typing import ClassVar

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....solver import DifferentialProblem, solve_diffrax
from ...electrophysiology._reaction import (
    _ExactFirstOrderGates,
    _FixedReactionLayout,
)


SOURCE_REVISION = "637da9ef28f7992e40fe79947364a51a38ec818c"
SOURCE_SHA256 = "e14e2aeffeb7b935017414a5ef53c06e43ed6b5fd4d7a92f07e0518b48b413c1"
SOURCE_URL = (
    "https://models.physiomeproject.org/workspace/"
    "shorten_ocallaghan_davidson_soboleva_2007/rawfile/"
    f"{SOURCE_REVISION}/shorten_ocallaghan_davidson_soboleva_2007.cellml"
)
SOURCE_LICENSE = "Creative Commons Attribution 3.0 Unported (CC BY 3.0)"
MODEL_ID = f"pmr-shorten-2007-fast-twitch:{SOURCE_REVISION}:{SOURCE_SHA256}"


def _symbols(component: str, names: tuple[str, ...], /) -> tuple[str, ...]:
    return tuple(f"{component}/{name}" for name in names)


_STATE_NAMES = (
    "vS",
    "vT",
    "K_t",
    "K_i",
    "K_e",
    "Na_i",
    "Na_t",
    "Na_e",
    "n",
    "h_K",
    "m",
    "h",
    "S",
    "n_t",
    "h_K_t",
    "m_t",
    "h_t",
    "S_t",
    "O_0",
    "O_1",
    "O_2",
    "O_3",
    "O_4",
    "C_0",
    "C_1",
    "C_2",
    "C_3",
    "C_4",
    "Ca_1",
    "Ca_SR1",
    "Ca_2",
    "Ca_SR2",
    "Ca_T_2",
    "Ca_P1",
    "Ca_P2",
    "Mg_P1",
    "Mg_P2",
    "Ca_Cs1",
    "Ca_Cs2",
    "Ca_ATP1",
    "Ca_ATP2",
    "Mg_ATP1",
    "Mg_ATP2",
    "ATP1",
    "ATP2",
    "Mg1",
    "Mg2",
    "Ca_CaT2",
    "D_0",
    "D_1",
    "D_2",
    "A_1",
    "A_2",
    "P",
    "P_SR",
    "P_C_SR",
)
_STATE_UNITS = (
    ("mV",) * 2
    + ("mM",) * 6
    + ("1",) * 20
    + ("uM",) * 25
    + ("mM",) * 3
)
_STATE_SYMBOLS = (
    _symbols("wal_environment", _STATE_NAMES[:8])
    + _symbols("sarco_DR_channel", _STATE_NAMES[8:10])
    + _symbols("sarco_Na_channel", _STATE_NAMES[10:13])
    + _symbols("t_DR_channel", _STATE_NAMES[13:15])
    + _symbols("t_Na_channel", _STATE_NAMES[15:18])
    + _symbols("sternrios", _STATE_NAMES[18:28])
    + _symbols("razumova", _STATE_NAMES[28:])
)
STATE_LAYOUT = _FixedReactionLayout(
    _STATE_NAMES, _STATE_UNITS, _STATE_SYMBOLS, "Shorten state"
)

_PARAMETER_NAMES = (
    "C_m",
    "gam",
    "R_a",
    "tsi",
    "tsi2",
    "tsi3",
    "FF",
    "tau_K",
    "tau_Na",
    "f_T",
    "tau_K2",
    "tau_Na2",
    "I_K_rest",
    "I_Na_rest",
    "alpha_h_bar",
    "alpha_m_bar",
    "alpha_n_bar",
    "beta_h_bar",
    "beta_m_bar",
    "beta_n_bar",
    "V_m",
    "V_n",
    "V_h",
    "V_a",
    "V_S_inf",
    "V_h_K_inf",
    "A_a",
    "A_S_inf",
    "A_h_K_inf",
    "K_alpha_h",
    "K_beta_h",
    "K_alpha_m",
    "K_alpha_n",
    "K_beta_m",
    "K_beta_n",
    "RR",
    "TT",
    "g_Cl_bar",
    "g_K_bar",
    "g_Na_bar",
    "G_K",
    "del",
    "K_K",
    "K_S",
    "K_m_K",
    "K_m_Na",
    "S_i",
    "J_NaK_bar",
    "V_tau",
    "eta_Cl",
    "eta_IR",
    "eta_DR",
    "eta_Na",
    "eta_NaK",
    "k_L",
    "k_Lm",
    "f",
    "alpha1",
    "K",
    "Vbar",
    "nu_SR",
    "K_SR",
    "L_e",
    "tau_R",
    "tau_SR_R",
    "L_x",
    "R_R",
    "k_T_on",
    "k_T_off",
    "T_tot",
    "k_P_on",
    "k_P_off",
    "P_tot",
    "k_Mg_on",
    "k_Mg_off",
    "k_Cs_on",
    "k_Cs_off",
    "Cs_tot",
    "k_CATP_on",
    "k_CATP_off",
    "k_MATP_on",
    "k_MATP_off",
    "tau_ATP",
    "tau_Mg",
    "k_0_on",
    "k_0_off",
    "k_Ca_on",
    "k_Ca_off",
    "f_o",
    "f_p",
    "h_o",
    "h_p",
    "g_o",
    "b_p",
    "k_p",
    "A_p",
    "B_p",
    "PP",
    "i2",
)
_PARAMETER_UNITS = (
    "uF/cm2",
    "1",
    "ohm cm2",
    "cm",
    "cm",
    "cm",
    "C/mol",
    "ms",
    "ms",
    "1",
    "ms",
    "ms",
    "uA/cm2",
    "uA/cm2",
    "1/ms",
    "1/(ms mV)",
    "1/(ms mV)",
    "1/ms",
    "1/ms",
    "1/ms",
    "mV",
    "mV",
    "mV",
    "mV",
    "mV",
    "mV",
    "mV",
    "mV",
    "mV",
    "mV",
    "mV",
    "mV",
    "mV",
    "mV",
    "mV",
    "mJ/(K mol)",
    "K",
    "mS/cm2",
    "mS/cm2",
    "mS/cm2",
    "mS/cm2",
    "1",
    "mM2",
    "mM2",
    "mM",
    "mM",
    "mM",
    "umol/(cm2 s)",
    "mV",
    "1",
    "1",
    "1",
    "1",
    "1",
    "1/ms",
    "1/ms",
    "1",
    "1/ms",
    "mV",
    "mV",
    "uM/(ms um3)",
    "uM",
    "um3/ms",
    "um3/ms",
    "um3/ms",
    "um",
    "um",
    "1/(uM ms)",
    "1/ms",
    "uM",
    "1/(uM ms)",
    "1/ms",
    "uM",
    "1/(uM ms)",
    "1/ms",
    "1/(uM ms)",
    "1/ms",
    "uM",
    "1/(uM ms)",
    "1/ms",
    "1/(uM ms)",
    "1/ms",
    "um3/ms",
    "um3/ms",
    "1/ms",
    "1/ms",
    "1/ms",
    "1/ms",
    "1/ms",
    "1/ms",
    "1/ms",
    "1/ms",
    "1/ms",
    "1/ms",
    "um3/ms",
    "1/(mM3 ms)",
    "1/(mM2 ms)",
    "mM2",
    "um3/ms",
)
_PARAMETER_SYMBOLS = (
    _symbols("wal_environment", _PARAMETER_NAMES[:54])
    + _symbols("sternrios", _PARAMETER_NAMES[54:60])
    + _symbols("razumova", _PARAMETER_NAMES[60:])
)
PARAMETER_LAYOUT = _FixedReactionLayout(
    _PARAMETER_NAMES,
    _PARAMETER_UNITS,
    _PARAMETER_SYMBOLS,
    "Shorten independent parameter",
)

_DERIVED_CONSTANT_NAMES = ("V_o", "V_SR", "V_1", "V_2", "V_SR1", "V_SR2")
_DERIVED_CONSTANT_UNITS = ("um3",) * 6
CONSTANT_LAYOUT = _FixedReactionLayout(
    _PARAMETER_NAMES + _DERIVED_CONSTANT_NAMES,
    _PARAMETER_UNITS + _DERIVED_CONSTANT_UNITS,
    _PARAMETER_SYMBOLS + _symbols("razumova", _DERIVED_CONSTANT_NAMES),
    "Shorten CellML constant",
)

_ALGEBRAIC_NAMES = (
    "I_T",
    "alpha_n",
    "h_K_inf",
    "alpha_h",
    "alpha_m",
    "S_inf",
    "alpha_n_t",
    "h_K_inf_t",
    "alpha_h_t",
    "alpha_m_t",
    "S_inf_t",
    "k_C",
    "T_0",
    "E_K",
    "beta_n",
    "tau_h_K",
    "beta_h",
    "beta_m",
    "tau_S",
    "beta_n_t",
    "tau_h_K_t",
    "beta_h_t",
    "beta_m_t",
    "tau_S_t",
    "k_Cm",
    "E_K_t",
    "Cl_i",
    "Cl_o",
    "Cl_i_t",
    "Cl_o_t",
    "J_K",
    "J_K_t",
    "I_HH",
    "a",
    "J_Cl",
    "g_Cl",
    "I_Cl",
    "K_R",
    "g_IR_bar",
    "y",
    "g_IR",
    "I_IR",
    "g_DR",
    "I_DR",
    "J_Na",
    "g_Na",
    "I_Na",
    "sig",
    "f1",
    "I_NaK_bar",
    "I_NaK",
    "I_ionic_s",
    "a_t",
    "J_Cl_t",
    "g_Cl_t",
    "I_Cl_t",
    "K_R_t",
    "g_IR_bar_t",
    "y_t",
    "g_IR_t",
    "I_IR_t",
    "g_DR_t",
    "I_DR_t",
    "J_Na_t",
    "g_Na_t",
    "I_Na_t",
    "sig_t",
    "f1_t",
    "I_NaK_bar_t",
    "I_NaK_t",
    "I_ionic_t",
)
_ALGEBRAIC_UNITS = (
    "uA/cm2",
    "1/ms",
    "1",
    "1/ms",
    "1/ms",
    "1",
    "1/ms",
    "1",
    "1/ms",
    "1/ms",
    "1",
    "1/ms",
    "uM",
    "mV",
    "1/ms",
    "ms",
    "1/ms",
    "1/ms",
    "ms",
    "1/ms",
    "ms",
    "1/ms",
    "1/ms",
    "ms",
    "1/ms",
    "mV",
    "mM",
    "mM",
    "mM",
    "mM",
    "mV mM",
    "mV mM",
    "uA/cm2",
    "1",
    "mV mM",
    "mS/cm2",
    "uA/cm2",
    "mM",
    "mS/cm2",
    "1",
    "mS/cm2",
    "uA/cm2",
    "mS/cm2",
    "uA/cm2",
    "mV mM",
    "mS/cm2",
    "uA/cm2",
    "1",
    "1",
    "uA/cm2",
    "uA/cm2",
    "uA/cm2",
    "1",
    "mV mM",
    "mS/cm2",
    "uA/cm2",
    "mM",
    "mS/cm2",
    "1",
    "mS/cm2",
    "uA/cm2",
    "mS/cm2",
    "uA/cm2",
    "mV mM",
    "mS/cm2",
    "uA/cm2",
    "1",
    "1",
    "uA/cm2",
    "uA/cm2",
    "uA/cm2",
)
_ALGEBRAIC_SYMBOLS = (
    _symbols("wal_environment", ("I_T",))
    + _symbols("sarco_DR_channel", ("alpha_n", "h_K_inf"))
    + _symbols("sarco_Na_channel", ("alpha_h", "alpha_m", "S_inf"))
    + _symbols("t_DR_channel", ("alpha_n_t", "h_K_inf_t"))
    + _symbols("t_Na_channel", ("alpha_h_t", "alpha_m_t", "S_inf_t"))
    + _symbols("sternrios", ("k_C",))
    + _symbols("razumova", ("T_0",))
    + _symbols("wal_environment", ("E_K",))
    + _symbols("sarco_DR_channel", ("beta_n", "tau_h_K"))
    + _symbols("sarco_Na_channel", ("beta_h", "beta_m", "tau_S"))
    + _symbols("t_DR_channel", ("beta_n_t", "tau_h_K_t"))
    + _symbols("t_Na_channel", ("beta_h_t", "beta_m_t", "tau_S_t"))
    + _symbols("sternrios", ("k_Cm",))
    + _symbols(
        "wal_environment",
        ("E_K_t", "Cl_i", "Cl_o", "Cl_i_t", "Cl_o_t", "J_K", "J_K_t"),
    )
    + _symbols("wal_environment", ("I_HH",))
    + _symbols("sarco_Cl_channel", ("a", "J_Cl", "g_Cl", "I_Cl"))
    + _symbols("sarco_IR_channel", ("K_R", "g_IR_bar", "y", "g_IR", "I_IR"))
    + _symbols("sarco_DR_channel", ("g_DR", "I_DR"))
    + _symbols("sarco_Na_channel", ("J_Na", "g_Na", "I_Na"))
    + _symbols("sarco_NaK_channel", ("sig", "f1", "I_NaK_bar", "I_NaK"))
    + _symbols("wal_environment", ("I_ionic_s",))
    + _symbols("t_Cl_channel", ("a_t", "J_Cl_t", "g_Cl_t", "I_Cl_t"))
    + _symbols("t_IR_channel", ("K_R_t", "g_IR_bar_t", "y_t", "g_IR_t", "I_IR_t"))
    + _symbols("t_DR_channel", ("g_DR_t", "I_DR_t"))
    + _symbols("t_Na_channel", ("J_Na_t", "g_Na_t", "I_Na_t"))
    + _symbols("t_NaK_channel", ("sig_t", "f1_t", "I_NaK_bar_t", "I_NaK_t"))
    + _symbols("wal_environment", ("I_ionic_t",))
)
ALGEBRAIC_LAYOUT = _FixedReactionLayout(
    _ALGEBRAIC_NAMES,
    _ALGEBRAIC_UNITS,
    _ALGEBRAIC_SYMBOLS,
    "Shorten algebraic",
)

_DEFAULT_PARAMETERS = jnp.asarray(
    (
        1.0,
        4.8,
        150.0,
        0.000001,
        0.0025,
        0.0005,
        96485.0,
        350.0,
        350.0,
        0.0032,
        21875.0,
        21875.0,
        1.02,
        -1.29,
        0.0081,
        0.288,
        0.0131,
        4.38,
        1.38,
        0.067,
        -46.0,
        -40.0,
        -45.0,
        70.0,
        -78.0,
        -40.0,
        150.0,
        5.8,
        7.5,
        14.7,
        9.0,
        10.0,
        7.0,
        18.0,
        40.0,
        8314.41,
        293.0,
        19.65,
        64.8,
        804.0,
        11.1,
        0.4,
        950.0,
        1.0,
        1.0,
        13.0,
        10.0,
        0.000621,
        90.0,
        0.1,
        1.0,
        0.45,
        0.1,
        0.1,
        0.002,
        1000.0,
        0.2,
        0.2,
        4.5,
        -20.0,
        4.875,
        1.0,
        0.00002,
        0.75,
        0.75,
        1.1,
        0.5,
        0.04425,
        0.115,
        140.0,
        0.0417,
        0.0005,
        1500.0,
        0.000033,
        0.003,
        0.000004,
        0.005,
        31000.0,
        0.15,
        30.0,
        0.0015,
        0.15,
        0.375,
        1.5,
        0.0,
        0.15,
        0.15,
        0.05,
        1.5,
        15.0,
        0.24,
        0.18,
        0.12,
        0.00002867,
        0.00000362,
        1.0,
        0.0001,
        6.0,
        300.0,
    )
)
_DEFAULT_STATE = jnp.asarray(
    (
        -79.974,
        -80.2,
        5.9,
        150.9,
        5.9,
        12.7,
        133.0,
        133.0,
        0.009466,
        0.9952,
        0.0358,
        0.4981,
        0.581,
        0.009466,
        0.9952,
        0.0358,
        0.4981,
        0.581,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.1,
        1500.0,
        0.1,
        1500.0,
        25.0,
        615.0,
        615.0,
        811.0,
        811.0,
        16900.0,
        16900.0,
        0.4,
        0.4,
        7200.0,
        7200.0,
        799.6,
        799.6,
        1000.0,
        1000.0,
        3.0,
        0.8,
        1.2,
        3.0,
        0.3,
        0.23,
        0.23,
        0.23,
        0.23,
    )
)

_GATE_INDICES = tuple(range(8, 18))
_EXACT_GATES = _ExactFirstOrderGates(
    _GATE_INDICES, substrate_id="shorten-2007-ten-voltage-gates-rush-larsen"
)

_POSITIVE_PARAMETER_INDICES = tuple(
    PARAMETER_LAYOUT.index(name)
    for name in (
        "C_m",
        "gam",
        "R_a",
        "tsi",
        "tsi2",
        "tsi3",
        "FF",
        "tau_K",
        "tau_Na",
        "tau_K2",
        "tau_Na2",
        "A_a",
        "A_S_inf",
        "A_h_K_inf",
        "K_alpha_h",
        "K_beta_h",
        "K_alpha_m",
        "K_alpha_n",
        "K_beta_m",
        "K_beta_n",
        "RR",
        "TT",
        "K_K",
        "K_S",
        "K_m_K",
        "K_m_Na",
        "S_i",
        "f",
        "K",
        "K_SR",
        "L_x",
        "R_R",
        "T_tot",
        "P_tot",
        "Cs_tot",
        "PP",
    )
)
_NONNEGATIVE_PARAMETER_INDICES = tuple(
    PARAMETER_LAYOUT.index(name)
    for name in (
        "f_T",
        "alpha_h_bar",
        "alpha_m_bar",
        "alpha_n_bar",
        "beta_h_bar",
        "beta_m_bar",
        "beta_n_bar",
        "g_Cl_bar",
        "g_K_bar",
        "g_Na_bar",
        "G_K",
        "J_NaK_bar",
        "eta_Cl",
        "eta_IR",
        "eta_DR",
        "eta_Na",
        "eta_NaK",
        "k_L",
        "k_Lm",
        "alpha1",
        "nu_SR",
        "L_e",
        "tau_R",
        "tau_SR_R",
        "k_T_on",
        "k_T_off",
        "k_P_on",
        "k_P_off",
        "k_Mg_on",
        "k_Mg_off",
        "k_Cs_on",
        "k_Cs_off",
        "k_CATP_on",
        "k_CATP_off",
        "k_MATP_on",
        "k_MATP_off",
        "tau_ATP",
        "tau_Mg",
        "k_0_on",
        "k_0_off",
        "k_Ca_on",
        "k_Ca_off",
        "f_o",
        "f_p",
        "h_o",
        "h_p",
        "g_o",
        "b_p",
        "k_p",
        "A_p",
        "B_p",
        "i2",
    )
)


def _parameters_admissible(parameters: ArrayLike, /) -> Array:
    values = PARAMETER_LAYOUT.require(parameters)
    delta = values[PARAMETER_LAYOUT.index("del")]
    return (
        jnp.all(jnp.isfinite(values))
        & jnp.all(values[jnp.asarray(_POSITIVE_PARAMETER_INDICES)] > 0.0)
        & jnp.all(values[jnp.asarray(_NONNEGATIVE_PARAMETER_INDICES)] >= 0.0)
        & (delta > 0.0)
        & (delta < 1.0)
    )




def _x_over_one_minus_exp_minus_x(value: Array, /) -> Array:
    """Stable ``x/(1-exp(-x))`` including its removable zero singularity."""
    small = jnp.abs(value) < 1.0e-4
    safe = jnp.where(small, jnp.ones_like(value), value)
    direct = safe / (-jnp.expm1(-safe))
    series = 1.0 + value / 2.0 + value**2 / 12.0 - value**4 / 720.0
    return jnp.where(small, series, direct)


def _ghk_drive(
    voltage_mV: Array,
    inside_mM: Array,
    outside_mM: Array,
    charge: float,
    faraday_C_per_mol: Array,
    gas_mJ_per_K_mol: Array,
    temperature_K: Array,
    /,
) -> Array:
    z = charge * faraday_C_per_mol * voltage_mV / (
        gas_mJ_per_K_mol * temperature_K
    )
    factor = gas_mJ_per_K_mol * temperature_K / (charge * faraday_C_per_mol)
    return factor * _x_over_one_minus_exp_minus_x(z) * (
        inside_mM - outside_mM * jnp.exp(-z)
    )


class ShortenPulseProtocol(StrictModule, NonTrainableState):
    """Finite, event-pinned rectangular inward-current pulse train."""

    onset_ms: Array
    width_ms: Array
    period_ms: Array
    amplitude_uA_per_cm2: Array
    pulse_count: int = eqx.field(static=True)
    protocol_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        onset_ms: ArrayLike = 0.0,
        width_ms: ArrayLike = 0.5,
        period_ms: ArrayLike = 50.0,
        amplitude_uA_per_cm2: ArrayLike = 150.0,
        pulse_count: int = 9,
    ):
        if not isinstance(pulse_count, int) or isinstance(pulse_count, bool):
            raise TypeError("pulse_count must be an integer.")
        if pulse_count <= 0:
            raise ValueError("pulse_count must be positive.")
        numeric = tuple(
            np.asarray(value)
            for value in (onset_ms, width_ms, period_ms, amplitude_uA_per_cm2)
        )
        if any(value.shape != () or not np.isfinite(value) for value in numeric):
            raise ValueError("Pulse protocol values must be finite real scalars.")
        onset, width, period, amplitude = (float(value) for value in numeric)
        if width <= 0.0 or period <= 0.0 or width >= period:
            raise ValueError("Pulse protocol requires 0 < width_ms < period_ms.")
        if amplitude < 0.0:
            raise ValueError("Pulse amplitude must be nonnegative (positive inward).")
        self.onset_ms = jnp.asarray(onset)
        self.width_ms = jnp.asarray(width)
        self.period_ms = jnp.asarray(period)
        self.amplitude_uA_per_cm2 = jnp.asarray(amplitude)
        self.pulse_count = pulse_count
        self.protocol_id = canonical_fingerprint(
            {
                "kind": "shorten-source-rectangular-pulses",
                "onset_ms": onset,
                "width_ms": width,
                "period_ms": period,
                "amplitude_uA_per_cm2": amplitude,
                "pulse_count": pulse_count,
                "endpoint": "left-closed-right-open",
            }
        )

    def current(self, time_ms: ArrayLike, /) -> Array:
        """Return the source-sign stimulus: positive current is inward."""
        time = jnp.asarray(time_ms)
        offset = time - self.onset_ms
        pulse = jnp.floor(offset / self.period_ms)
        phase = offset - pulse * self.period_ms
        active = (
            (offset >= 0.0)
            & (pulse < self.pulse_count)
            & (phase >= 0.0)
            & (phase < self.width_ms)
        )
        return jnp.where(active, self.amplitude_uA_per_cm2, 0.0)

    def event_times_ms(self) -> Array:
        starts = self.onset_ms + self.period_ms * jnp.arange(self.pulse_count)
        return jnp.stack((starts, starts + self.width_ms), axis=-1).reshape(-1)

_DEFAULT_PULSE_PROTOCOL = ShortenPulseProtocol()


class ShortenFastTwitchEvaluation(StrictModule):
    """Pure source-layout rates, algebraics, currents, calcium, and tension data."""

    state_rate_per_ms: Array
    algebraic: Array
    constants: Array
    gate_steady_state: Array
    gate_time_constant_ms: Array
    sarcolemmal_current_uA_per_cm2: Array
    tubular_current_uA_per_cm2: Array
    axial_current_uA_per_cm2: Array
    stimulus_current_uA_per_cm2: Array
    cytosolic_calcium_uM: Array
    sr_calcium_uM: Array
    calcium_release_open_probability: Array
    force_bearing_crossbridge_uM: Array
    tension_driver_uM: Array
    phosphate_mM: Array
    valid: Array
    current_names: tuple[str, ...] = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def algebraic_value(self, name: str, /) -> Array:
        return self.algebraic[..., ALGEBRAIC_LAYOUT.index(name)]

    def current(self, membrane: str, name: str, /) -> Array:
        if name not in self.current_names:
            raise KeyError(f"Unknown Shorten membrane current {name!r}.")
        index = self.current_names.index(name)
        if membrane == "sarcolemma":
            return self.sarcolemmal_current_uA_per_cm2[..., index]
        if membrane == "t_tubule":
            return self.tubular_current_uA_per_cm2[..., index]
        raise KeyError("membrane must be 'sarcolemma' or 't_tubule'.")


class ShortenFastTwitchModel(StrictModule):
    """Complete 56-state fast-twitch CellML model with dynamic parameters."""

    parameters: Array
    model_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    source_sha256: str = eqx.field(static=True)
    source_url: str = eqx.field(static=True)
    source_license: str = eqx.field(static=True)

    state_layout: ClassVar[_FixedReactionLayout] = STATE_LAYOUT
    parameter_layout: ClassVar[_FixedReactionLayout] = PARAMETER_LAYOUT
    constant_layout: ClassVar[_FixedReactionLayout] = CONSTANT_LAYOUT
    algebraic_layout: ClassVar[_FixedReactionLayout] = ALGEBRAIC_LAYOUT
    current_names: ClassVar[tuple[str, ...]] = (
        "chloride",
        "inward_rectifier_potassium",
        "delayed_rectifier_potassium",
        "sodium",
        "sodium_potassium_pump",
    )

    def __init__(self, parameters: ArrayLike | None = None):
        values = _DEFAULT_PARAMETERS if parameters is None else jnp.asarray(parameters)
        values = PARAMETER_LAYOUT.require(values)
        if values.shape != (PARAMETER_LAYOUT.count,):
            raise ValueError("Shorten model parameters must be one unbatched vector.")
        if not jnp.issubdtype(values.dtype, jnp.floating):
            raise TypeError("Shorten model parameters must have floating dtype.")
        if not bool(np.asarray(_parameters_admissible(values))):
            raise ValueError(
                "Shorten model parameters violate source positivity or range bounds."
            )
        self.parameters = values
        self.model_id = MODEL_ID
        self.source_revision = SOURCE_REVISION
        self.source_sha256 = SOURCE_SHA256
        self.source_url = SOURCE_URL
        self.source_license = SOURCE_LICENSE

    @property
    def membrane_capacitance_uF_per_cm2(self) -> Array:
        return self.parameters[0]

    def initialize(
        self,
        batch_shape: tuple[int, ...] = (),
        *,
        dtype: object | None = None,
    ) -> Array:
        shape = tuple(int(size) for size in batch_shape)
        if any(size < 0 for size in shape):
            raise ValueError("batch_shape entries must be nonnegative.")
        resolved_dtype = self.parameters.dtype if dtype is None else jnp.dtype(dtype)
        if not jnp.issubdtype(resolved_dtype, jnp.floating):
            raise TypeError("Shorten state dtype must be floating point.")
        return jnp.broadcast_to(_DEFAULT_STATE.astype(resolved_dtype), shape + (56,))

    def constants(self) -> Array:
        p = self.parameters
        length = p[65]
        radius = p[66]
        volume = 0.95 * length * jnp.pi * radius**2
        sr_volume = 0.05 * length * jnp.pi * radius**2
        return jnp.concatenate(
            (
                p,
                jnp.stack(
                    (
                        volume,
                        sr_volume,
                        0.01 * volume,
                        0.99 * volume,
                        0.01 * sr_volume,
                        0.99 * sr_volume,
                    )
                ),
            )
        )

    def evaluate(
        self,
        time_ms: ArrayLike,
        state: ArrayLike,
        /,
        *,
        protocol: ShortenPulseProtocol | None = None,
        stimulus_current_uA_per_cm2: ArrayLike | None = None,
    ) -> ShortenFastTwitchEvaluation:
        """Evaluate all 56 RHS entries and 71 source algebraics without mutation."""
        y = STATE_LAYOUT.require(state)
        if not jnp.issubdtype(y.dtype, jnp.floating):
            raise TypeError("Shorten state must have floating dtype.")
        time = jnp.asarray(time_ms, dtype=y.dtype)
        if time.shape not in ((), y.shape[:-1]):
            raise ValueError("time_ms must be scalar or match the state batch axes.")
        p = self.parameters.astype(y.dtype)
        c = self.constants().astype(y.dtype)

        (
            v_s,
            v_t,
            k_t,
            k_i,
            k_e,
            na_i,
            na_t,
            na_e,
            n,
            h_k,
            m_gate,
            h_gate,
            s_gate,
            n_t,
            h_k_t,
            m_t,
            h_t,
            s_t,
            o0,
            o1,
            o2,
            o3,
            o4,
            c0,
            c1,
            c2,
            c3,
            c4,
            ca1,
            ca_sr1,
            ca2,
            ca_sr2,
            ca_t2,
            ca_p1,
            ca_p2,
            mg_p1,
            mg_p2,
            ca_cs1,
            ca_cs2,
            ca_atp1,
            ca_atp2,
            mg_atp1,
            mg_atp2,
            atp1,
            atp2,
            mg1,
            mg2,
            ca_cat2,
            d0,
            d1,
            d2,
            a1,
            a2,
            phosphate,
            phosphate_sr,
            phosphate_complex_sr,
        ) = tuple(y[..., index] for index in range(56))

        (
            capacitance,
            gamma,
            axial_resistance,
            t_shell,
            s_shell,
            e_shell,
            faraday,
            tau_k,
            tau_na,
            f_t,
            tau_k2,
            tau_na2,
            i_k_rest,
            i_na_rest,
            alpha_h_bar,
            alpha_m_bar,
            alpha_n_bar,
            beta_h_bar,
            beta_m_bar,
            beta_n_bar,
            v_m,
            v_n,
            v_h,
            v_a,
            v_s_inf,
            v_hk_inf,
            a_a,
            a_s_inf,
            a_hk_inf,
            k_alpha_h,
            k_beta_h,
            k_alpha_m,
            k_alpha_n,
            k_beta_m,
            k_beta_n,
            gas,
            temperature,
            g_cl_bar,
            g_k_bar,
            g_na_bar,
            g_k_ir,
            delta,
            k_k,
            k_s,
            km_k,
            km_na,
            s_i,
            j_nak_bar,
            v_tau,
            eta_cl,
            eta_ir,
            eta_dr,
            eta_na,
            eta_nak,
            k_l,
            k_lm,
            f_ratio,
            alpha1,
            k_voltage,
            v_bar,
            nu_sr,
            k_sr,
            leak,
            tau_r,
            tau_sr_r,
            length,
            radius,
            k_t_on,
            k_t_off,
            t_total,
            k_p_on,
            k_p_off,
            p_total,
            k_mg_on,
            k_mg_off,
            k_cs_on,
            k_cs_off,
            cs_total,
            k_catp_on,
            k_catp_off,
            k_matp_on,
            k_matp_off,
            tau_atp,
            tau_mg,
            k0_on,
            k0_off,
            kca_on,
            kca_off,
            f_o,
            f_p,
            h_o,
            h_p,
            g_o,
            b_p,
            k_p_transport,
            a_precip,
            b_precip,
            precip_product,
            release_scale,
        ) = tuple(p[index] for index in range(99))
        del length, radius
        volume, sr_volume, volume1, volume2, sr_volume1, sr_volume2 = (
            c[index] for index in range(99, 105)
        )
        del volume, sr_volume

        if stimulus_current_uA_per_cm2 is None:
            selected_protocol = (
                _DEFAULT_PULSE_PROTOCOL if protocol is None else protocol
            )
            if not isinstance(selected_protocol, ShortenPulseProtocol):
                raise TypeError("protocol must be a ShortenPulseProtocol or None.")
            stimulus = selected_protocol.current(time)
        else:
            if protocol is not None:
                raise ValueError(
                    "Specify either protocol or stimulus_current_uA_per_cm2, not both."
                )
            stimulus = jnp.asarray(stimulus_current_uA_per_cm2, dtype=y.dtype)
        stimulus = jnp.broadcast_to(stimulus, y.shape[:-1])

        alpha_n = (
            alpha_n_bar
            * k_alpha_n
            * _x_over_one_minus_exp_minus_x((v_s - v_n) / k_alpha_n)
        )
        beta_n = beta_n_bar * jnp.exp(-(v_s - v_n) / k_beta_n)
        h_k_inf = 1.0 / (1.0 + jnp.exp((v_s - v_hk_inf) / a_hk_inf))
        tau_h_k = 1000.0 * jnp.exp(-(v_s + 40.0) / 25.75)
        alpha_m = (
            alpha_m_bar
            * k_alpha_m
            * _x_over_one_minus_exp_minus_x((v_s - v_m) / k_alpha_m)
        )
        beta_m = beta_m_bar * jnp.exp(-(v_s - v_m) / k_beta_m)
        alpha_h = alpha_h_bar * jnp.exp(-(v_s - v_h) / k_alpha_h)
        beta_h = beta_h_bar / (1.0 + jnp.exp(-(v_s - v_h) / k_beta_h))
        s_inf = 1.0 / (1.0 + jnp.exp((v_s - v_s_inf) / a_s_inf))
        tau_s = 8571.0 / (0.2 + 5.65 * ((v_s + v_tau) / 100.0) ** 2)

        alpha_n_t = (
            alpha_n_bar
            * k_alpha_n
            * _x_over_one_minus_exp_minus_x((v_t - v_n) / k_alpha_n)
        )
        beta_n_t = beta_n_bar * jnp.exp(-(v_t - v_n) / k_beta_n)
        h_k_inf_t = 1.0 / (1.0 + jnp.exp((v_t - v_hk_inf) / a_hk_inf))
        tau_h_k_t = jnp.exp(-(v_t + 40.0) / 25.75)
        alpha_m_t = (
            alpha_m_bar
            * k_alpha_m
            * _x_over_one_minus_exp_minus_x((v_t - v_m) / k_alpha_m)
        )
        beta_m_t = beta_m_bar * jnp.exp(-(v_t - v_m) / k_beta_m)
        alpha_h_t = alpha_h_bar * jnp.exp(-(v_t - v_h) / k_alpha_h)
        beta_h_t = beta_h_bar / (1.0 + jnp.exp(-(v_t - v_h) / k_beta_h))
        s_inf_t = 1.0 / (1.0 + jnp.exp((v_t - v_s_inf) / a_s_inf))
        tau_s_t = 8571.0 / (0.2 + 5.65 * ((v_t + v_tau) / 100.0) ** 2)

        gate_steady = jnp.stack(
            (
                alpha_n / (alpha_n + beta_n),
                h_k_inf,
                alpha_m / (alpha_m + beta_m),
                alpha_h / (alpha_h + beta_h),
                s_inf,
                alpha_n_t / (alpha_n_t + beta_n_t),
                h_k_inf_t,
                alpha_m_t / (alpha_m_t + beta_m_t),
                alpha_h_t / (alpha_h_t + beta_h_t),
                s_inf_t,
            ),
            axis=-1,
        )
        gate_tau = jnp.stack(
            (
                1.0 / (alpha_n + beta_n),
                tau_h_k,
                1.0 / (alpha_m + beta_m),
                1.0 / (alpha_h + beta_h),
                tau_s,
                1.0 / (alpha_n_t + beta_n_t),
                tau_h_k_t,
                1.0 / (alpha_m_t + beta_m_t),
                1.0 / (alpha_h_t + beta_h_t),
                tau_s_t,
            ),
            axis=-1,
        )

        k_c = 0.5 * alpha1 * jnp.exp((v_t - v_bar) / (8.0 * k_voltage))
        k_cm = 0.5 * alpha1 * jnp.exp((v_bar - v_t) / (8.0 * k_voltage))
        open_probability = o0 + o1 + o2 + o3 + o4

        t0 = t_total - ca_t2 - ca_cat2 - d0 - d1 - d2 - a1 - a2
        ca_p_binding1 = k_p_on * ca1 * (p_total - ca_p1 - mg_p1) - k_p_off * ca_p1
        ca_p_binding2 = k_p_on * ca2 * (p_total - ca_p2 - mg_p2) - k_p_off * ca_p2
        mg_p_binding1 = k_mg_on * (p_total - ca_p1 - mg_p1) * mg1 - k_mg_off * mg_p1
        mg_p_binding2 = k_mg_on * (p_total - ca_p2 - mg_p2) * mg2 - k_mg_off * mg_p2
        cs_binding1 = k_cs_on * ca_sr1 * (cs_total - ca_cs1) - k_cs_off * ca_cs1
        cs_binding2 = k_cs_on * ca_sr2 * (cs_total - ca_cs2) - k_cs_off * ca_cs2
        ca_atp_binding1 = k_catp_on * ca1 * atp1 - k_catp_off * ca_atp1
        ca_atp_binding2 = k_catp_on * ca2 * atp2 - k_catp_off * ca_atp2
        mg_atp_binding1 = k_matp_on * mg1 * atp1 - k_matp_off * mg_atp1
        mg_atp_binding2 = k_matp_on * mg2 * atp2 - k_matp_off * mg_atp2

        saturation = 0.001 * phosphate_sr * ca_sr2 - precip_product
        positive_saturation = jnp.where(saturation > 0.0, saturation, 0.0)
        negative_saturation = jnp.where(saturation < 0.0, -saturation, 0.0)
        precipitation = (
            a_precip
            * positive_saturation
            * 0.001
            * phosphate_sr
            * ca_sr2
            - b_precip * phosphate_complex_sr * negative_saturation
        )

        rate_ca1 = (
            release_scale * open_probability * (ca_sr1 - ca1) / volume1
            - nu_sr * (ca1 / (ca1 + k_sr)) / volume1
            + leak * (ca_sr1 - ca1) / volume1
            - tau_r * (ca1 - ca2) / volume1
            - ca_p_binding1
            - ca_atp_binding1
        )
        rate_ca_sr1 = (
            -release_scale * open_probability * (ca_sr1 - ca1) / sr_volume1
            + nu_sr * (ca1 / (ca1 + k_sr)) / sr_volume1
            - leak * (ca_sr1 - ca1) / sr_volume1
            - tau_sr_r * (ca_sr1 - ca_sr2) / sr_volume1
            - cs_binding1
        )
        rate_ca_sr2 = (
            nu_sr * (ca2 / (ca2 + k_sr)) / sr_volume2
            - leak * (ca_sr2 - ca2) / sr_volume2
            + tau_sr_r * (ca_sr1 - ca_sr2) / sr_volume2
            - cs_binding2
            - 1000.0 * precipitation
        )

        binding_t0_ca = k_t_on * ca2 * t0 - k_t_off * ca_t2
        binding_t2_ca = k_t_on * ca2 * ca_t2 - k_t_off * ca_cat2
        binding_d0_ca = k_t_on * ca2 * d0 - k_t_off * d1
        binding_d1_ca = k_t_on * ca2 * d1 - k_t_off * d2
        calcium_troponin_flux = (
            binding_t0_ca + binding_t2_ca + binding_d0_ca + binding_d1_ca
        )
        rate_ca2 = (
            -nu_sr * (ca2 / (ca2 + k_sr)) / volume2
            + leak * (ca_sr2 - ca2) / volume2
            + tau_r * (ca1 - ca2) / volume2
            - calcium_troponin_flux
            - ca_p_binding2
            - ca_atp_binding2
        )
        rate_ca_t2 = binding_t0_ca - binding_t2_ca - k0_on * ca_t2 + k0_off * d1
        rate_ca_cat2 = binding_t2_ca - kca_on * ca_cat2 + kca_off * d2
        rate_d0 = -binding_d0_ca + k0_on * t0 - k0_off * d0
        rate_d1 = (
            binding_d0_ca + k0_on * ca_t2 - k0_off * d1 - binding_d1_ca
        )
        rate_d2 = (
            binding_d1_ca
            + kca_on * ca_cat2
            - kca_off * d2
            - f_o * d2
            + f_p * a1
            + g_o * a2
        )
        rate_a1 = f_o * d2 - f_p * a1 + h_p * a2 - h_o * a1
        rate_a2 = -h_p * a2 + h_o * a1 - g_o * a2

        rate_p = (
            0.001 * (h_o * a1 - h_p * a2)
            - b_p * phosphate
            - k_p_transport * (phosphate - phosphate_sr) / volume2
        )
        rate_p_sr = (
            k_p_transport * (phosphate - phosphate_sr) / sr_volume2
            - precipitation
        )
        rate_p_complex_sr = precipitation

        rate_ca_p1 = ca_p_binding1
        rate_ca_p2 = ca_p_binding2
        rate_mg_p1 = mg_p_binding1
        rate_mg_p2 = mg_p_binding2
        rate_ca_cs1 = cs_binding1
        rate_ca_cs2 = cs_binding2
        rate_ca_atp1 = ca_atp_binding1 - tau_atp * (ca_atp1 - ca_atp2) / volume1
        rate_ca_atp2 = ca_atp_binding2 + tau_atp * (ca_atp1 - ca_atp2) / volume2
        rate_mg_atp1 = mg_atp_binding1 - tau_atp * (mg_atp1 - mg_atp2) / volume1
        rate_mg_atp2 = mg_atp_binding2 + tau_atp * (mg_atp1 - mg_atp2) / volume2
        rate_atp1 = (
            -ca_atp_binding1
            - mg_atp_binding1
            - tau_atp * (atp1 - atp2) / volume1
        )
        rate_atp2 = (
            -ca_atp_binding2
            - mg_atp_binding2
            + tau_atp * (atp1 - atp2) / volume2
        )
        rate_mg1 = (
            -mg_p_binding1
            - mg_atp_binding1
            - tau_mg * (mg1 - mg2) / volume1
        )
        rate_mg2 = (
            -mg_p_binding2
            - mg_atp_binding2
            + tau_mg * (mg1 - mg2) / volume2
        )

        rate_c0 = -k_l * c0 + k_lm * o0 - 4.0 * k_c * c0 + k_cm * c1
        rate_o0 = (
            k_l * c0
            - k_lm * o0
            - 4.0 * k_c * o0 / f_ratio
            + f_ratio * k_cm * o1
        )
        rate_c1 = (
            4.0 * k_c * c0
            - k_cm * c1
            - k_l * c1 / f_ratio
            + f_ratio * k_lm * o1
            - 3.0 * k_c * c1
            + 2.0 * k_cm * c2
        )
        rate_o1 = (
            k_l * c1 / f_ratio
            - k_lm * f_ratio * o1
            + 4.0 * k_c * o0 / f_ratio
            - f_ratio * k_cm * o1
            - 3.0 * k_c * o1 / f_ratio
            + 2.0 * f_ratio * k_cm * o2
        )
        rate_c2 = (
            3.0 * k_c * c1
            - 2.0 * k_cm * c2
            - k_l * c2 / f_ratio**2
            + f_ratio**2 * k_lm * o2
            - 2.0 * k_c * c2
            + 3.0 * k_cm * c3
        )
        rate_o2 = (
            3.0 * k_c * o1 / f_ratio
            - 2.0 * f_ratio * k_cm * o2
            + k_l * c2 / f_ratio**2
            - k_lm * f_ratio**2 * o2
            - 2.0 * k_c * o2 / f_ratio
            + 3.0 * f_ratio * k_cm * o3
        )
        rate_c3 = (
            2.0 * k_c * c2
            - 3.0 * k_cm * c3
            - k_l * c3 / f_ratio**3
            + k_lm * f_ratio**3 * o3
            - k_c * c3
            + 4.0 * k_cm * c4
        )
        rate_o3 = (
            k_l * c3 / f_ratio**3
            - k_lm * f_ratio**3 * o3
            + 2.0 * k_c * o2 / f_ratio
            - 3.0 * k_cm * f_ratio * o3
            - k_c * o3 / f_ratio
            + 4.0 * f_ratio * k_cm * o4
        )
        rate_c4 = (
            k_c * c3
            - 4.0 * k_cm * c4
            - k_l * c4 / f_ratio**4
            + k_lm * f_ratio**4 * o4
        )
        rate_o4 = (
            k_c * o3 / f_ratio
            - 4.0 * f_ratio * k_cm * o4
            + k_l * c4 / f_ratio**4
            - k_lm * f_ratio**4 * o4
        )

        j_k = _ghk_drive(v_s, k_i, k_e, 1.0, faraday, gas, temperature)
        e_k = gas * temperature / faraday * jnp.log(k_e / k_i)
        k_r = k_e * jnp.exp(-delta * e_k * faraday / (gas * temperature))
        g_ir_bar = g_k_ir * k_r**2 / (k_k + k_r**2)
        y_ir = 1.0 - 1.0 / (
            1.0
            + k_s
            * (1.0 + k_r**2 / k_k)
            / (
                s_i**2
                * jnp.exp(
                    2.0
                    * (1.0 - delta)
                    * v_s
                    * faraday
                    / (gas * temperature)
                )
            )
        )
        g_ir = g_ir_bar * y_ir
        i_ir = g_ir * jnp.where(j_k > 0.0, j_k / 50.0, 0.0)
        g_dr = g_k_bar * n**4 * h_k
        i_dr = g_dr * j_k / 50.0
        j_na = _ghk_drive(v_s, na_i, na_e, 1.0, faraday, gas, temperature)
        g_na = g_na_bar * m_gate**3 * h_gate * s_gate
        i_na = g_na * j_na / 75.0
        sigma = (jnp.exp(na_e / 67.3) - 1.0) / 7.0
        pump_voltage = 1.0 / (
            1.0
            + 0.12 * jnp.exp(-0.1 * v_s * faraday / (gas * temperature))
            + 0.04 * sigma * jnp.exp(-v_s * faraday / (gas * temperature))
        )
        i_nak_bar = faraday * j_nak_bar / (
            (1.0 + km_k / k_e) ** 2 * (1.0 + km_na / na_i) ** 3
        )
        i_nak = i_nak_bar * pump_voltage
        cl_i = 156.5 / (5.0 + jnp.exp(-faraday * e_k / (gas * temperature)))
        cl_o = 156.5 - 5.0 * cl_i
        j_cl = _ghk_drive(v_s, cl_i, cl_o, -1.0, faraday, gas, temperature)
        a_cl = 1.0 / (1.0 + jnp.exp((v_s - v_a) / a_a))
        g_cl = g_cl_bar * a_cl**4
        i_cl = g_cl * j_cl / 45.0

        j_k_t = _ghk_drive(v_t, k_i, k_t, 1.0, faraday, gas, temperature)
        e_k_t = gas * temperature / faraday * jnp.log(k_t / k_i)
        k_r_t = k_t * jnp.exp(-delta * e_k_t * faraday / (gas * temperature))
        g_ir_bar_t = g_k_ir * k_r_t**2 / (k_k + k_r_t**2)
        y_ir_t = 1.0 - 1.0 / (
            1.0
            + k_s
            * (1.0 + k_r_t**2 / k_k)
            / (
                s_i**2
                * jnp.exp(
                    2.0
                    * (1.0 - delta)
                    * v_t
                    * faraday
                    / (gas * temperature)
                )
            )
        )
        g_ir_t = g_ir_bar_t * y_ir_t
        i_ir_t = eta_ir * g_ir_t * j_k_t / 50.0
        g_dr_t = g_k_bar * n_t**4 * h_k_t
        i_dr_t = eta_dr * g_dr_t * j_k_t / 50.0
        j_na_t = _ghk_drive(v_t, na_i, na_t, 1.0, faraday, gas, temperature)
        g_na_t = g_na_bar * m_t**3 * h_t * s_t
        i_na_t = eta_na * g_na_t * j_na_t / 75.0
        sigma_t = (jnp.exp(na_t / 67.3) - 1.0) / 7.0
        pump_voltage_t = 1.0 / (
            1.0
            + 0.12 * jnp.exp(-0.1 * v_t * faraday / (gas * temperature))
            + 0.04 * sigma_t * jnp.exp(-v_t * faraday / (gas * temperature))
        )
        i_nak_bar_t = faraday * j_nak_bar / (
            (1.0 + km_k / k_t) ** 2 * (1.0 + km_na / na_i) ** 3
        )
        i_nak_t = eta_nak * i_nak_bar_t * pump_voltage_t
        cl_i_t = 156.5 / (5.0 + jnp.exp(-faraday * e_k_t / (gas * temperature)))
        cl_o_t = 156.5 - 5.0 * cl_i_t
        j_cl_t = _ghk_drive(v_t, cl_i_t, cl_o_t, -1.0, faraday, gas, temperature)
        a_cl_t = 1.0 / (1.0 + jnp.exp((v_t - v_a) / a_a))
        g_cl_t = g_cl_bar * a_cl_t**4
        i_cl_t = eta_cl * g_cl_t * j_cl_t / 45.0

        axial_current = 1000.0 * (v_s - v_t) / axial_resistance
        ionic_s = i_cl + i_ir + i_dr + i_na + i_nak - stimulus
        ionic_t = i_cl_t + i_ir_t + i_dr_t + i_na_t + i_nak_t
        rate_vs = -(ionic_s + axial_current) / capacitance
        rate_vt = -(ionic_t - axial_current / gamma) / capacitance
        rate_ke = (
            (i_ir + i_dr + i_k_rest - 2.0 * i_nak)
            / (1000.0 * faraday * e_shell)
            + (k_t - k_e) / tau_k2
        )
        rate_nae = (
            (i_na + i_na_rest + 3.0 * i_nak)
            / (1000.0 * faraday * e_shell)
            + (na_t - na_e) / tau_na2
        )
        rate_ki = (
            -f_t
            * (i_ir_t + i_dr_t + i_k_rest - 2.0 * i_nak_t)
            / (1000.0 * faraday * t_shell)
            - (i_ir + i_dr + i_k_rest - 2.0 * i_nak)
            / (1000.0 * faraday * s_shell)
        )
        rate_kt = (
            (i_ir_t + i_dr_t + i_k_rest - 2.0 * i_nak_t)
            / (1000.0 * faraday * t_shell)
            - (k_t - k_e) / tau_k
        )
        rate_nai = (
            -f_t
            * (i_na_t + i_na_rest + 3.0 * i_nak_t)
            / (1000.0 * faraday * t_shell)
            - (i_na + i_na_rest + 3.0 * i_nak)
            / (1000.0 * faraday * s_shell)
        )
        rate_nat = (
            (i_na_t + i_na_rest + 3.0 * i_nak_t)
            / (1000.0 * faraday * t_shell)
            - (na_t - na_e) / tau_na
        )

        rate_n = alpha_n * (1.0 - n) - beta_n * n
        rate_hk = (h_k_inf - h_k) / tau_h_k
        rate_m = alpha_m * (1.0 - m_gate) - beta_m * m_gate
        rate_h = alpha_h * (1.0 - h_gate) - beta_h * h_gate
        rate_s = (s_inf - s_gate) / tau_s
        rate_nt = alpha_n_t * (1.0 - n_t) - beta_n_t * n_t
        rate_hkt = (h_k_inf_t - h_k_t) / tau_h_k_t
        rate_mt = alpha_m_t * (1.0 - m_t) - beta_m_t * m_t
        rate_ht = alpha_h_t * (1.0 - h_t) - beta_h_t * h_t
        rate_st = (s_inf_t - s_t) / tau_s_t

        rates = jnp.stack(
            (
                rate_vs,
                rate_vt,
                rate_kt,
                rate_ki,
                rate_ke,
                rate_nai,
                rate_nat,
                rate_nae,
                rate_n,
                rate_hk,
                rate_m,
                rate_h,
                rate_s,
                rate_nt,
                rate_hkt,
                rate_mt,
                rate_ht,
                rate_st,
                rate_o0,
                rate_o1,
                rate_o2,
                rate_o3,
                rate_o4,
                rate_c0,
                rate_c1,
                rate_c2,
                rate_c3,
                rate_c4,
                rate_ca1,
                rate_ca_sr1,
                rate_ca2,
                rate_ca_sr2,
                rate_ca_t2,
                rate_ca_p1,
                rate_ca_p2,
                rate_mg_p1,
                rate_mg_p2,
                rate_ca_cs1,
                rate_ca_cs2,
                rate_ca_atp1,
                rate_ca_atp2,
                rate_mg_atp1,
                rate_mg_atp2,
                rate_atp1,
                rate_atp2,
                rate_mg1,
                rate_mg2,
                rate_ca_cat2,
                rate_d0,
                rate_d1,
                rate_d2,
                rate_a1,
                rate_a2,
                rate_p,
                rate_p_sr,
                rate_p_complex_sr,
            ),
            axis=-1,
        )
        algebraic = jnp.stack(
            (
                axial_current,
                alpha_n,
                h_k_inf,
                alpha_h,
                alpha_m,
                s_inf,
                alpha_n_t,
                h_k_inf_t,
                alpha_h_t,
                alpha_m_t,
                s_inf_t,
                k_c,
                t0,
                e_k,
                beta_n,
                tau_h_k,
                beta_h,
                beta_m,
                tau_s,
                beta_n_t,
                tau_h_k_t,
                beta_h_t,
                beta_m_t,
                tau_s_t,
                k_cm,
                e_k_t,
                cl_i,
                cl_o,
                cl_i_t,
                cl_o_t,
                j_k,
                j_k_t,
                stimulus,
                a_cl,
                j_cl,
                g_cl,
                i_cl,
                k_r,
                g_ir_bar,
                y_ir,
                g_ir,
                i_ir,
                g_dr,
                i_dr,
                j_na,
                g_na,
                i_na,
                sigma,
                pump_voltage,
                i_nak_bar,
                i_nak,
                ionic_s,
                a_cl_t,
                j_cl_t,
                g_cl_t,
                i_cl_t,
                k_r_t,
                g_ir_bar_t,
                y_ir_t,
                g_ir_t,
                i_ir_t,
                g_dr_t,
                i_dr_t,
                j_na_t,
                g_na_t,
                i_na_t,
                sigma_t,
                pump_voltage_t,
                i_nak_bar_t,
                i_nak_t,
                ionic_t,
            ),
            axis=-1,
        )
        sarco_currents = jnp.stack((i_cl, i_ir, i_dr, i_na, i_nak), axis=-1)
        tubular_currents = jnp.stack(
            (i_cl_t, i_ir_t, i_dr_t, i_na_t, i_nak_t), axis=-1
        )
        valid = self.admissible(y, precomputed_rates=rates, t0=t0)
        return ShortenFastTwitchEvaluation(
            rates,
            algebraic,
            c,
            gate_steady,
            gate_tau,
            sarco_currents,
            tubular_currents,
            axial_current,
            stimulus,
            jnp.stack((ca1, ca2), axis=-1),
            jnp.stack((ca_sr1, ca_sr2), axis=-1),
            open_probability,
            a2,
            a2,
            jnp.stack((phosphate, phosphate_sr, phosphate_complex_sr), axis=-1),
            valid,
            self.current_names,
            self.model_id,
        )

    def rhs(
        self,
        time_ms: ArrayLike,
        state: ArrayLike,
        /,
        *,
        protocol: ShortenPulseProtocol | None = None,
        stimulus_current_uA_per_cm2: ArrayLike | None = None,
    ) -> Array:
        return self.evaluate(
            time_ms,
            state,
            protocol=protocol,
            stimulus_current_uA_per_cm2=stimulus_current_uA_per_cm2,
        ).state_rate_per_ms

    def exact_gate_update(
        self,
        time_ms: ArrayLike,
        state: ArrayLike,
        dt_ms: ArrayLike,
        /,
    ) -> Array:
        evaluation = self.evaluate(
            time_ms, state, stimulus_current_uA_per_cm2=0.0
        )
        return _EXACT_GATES.update(
            state,
            evaluation.gate_steady_state,
            evaluation.gate_time_constant_ms,
            dt_ms,
        )

    def admissible(
        self,
        state: ArrayLike,
        /,
        *,
        precomputed_rates: ArrayLike | None = None,
        t0: ArrayLike | None = None,
    ) -> Array:
        y = STATE_LAYOUT.require(state)
        finite = jnp.all(jnp.isfinite(y), axis=-1)
        if precomputed_rates is not None:
            finite = finite & jnp.all(
                jnp.isfinite(jnp.asarray(precomputed_rates)), axis=-1
            )
        parameter_valid = _parameters_admissible(self.parameters)
        positive_ions = jnp.all(y[..., 2:8] > 0.0, axis=-1)
        gates = jnp.all((y[..., 8:18] >= 0.0) & (y[..., 8:18] <= 1.0), axis=-1)
        ryr = y[..., 18:28]
        ryr_probability = jnp.all(ryr >= -1.0e-8, axis=-1) & (
            jnp.abs(jnp.sum(ryr, axis=-1) - 1.0) <= 1.0e-4
        )
        nonnegative_pools = jnp.all(y[..., 28:] >= -1.0e-8, axis=-1)
        free_troponin = (
            self.parameters[69]
            - y[..., 32]
            - jnp.sum(y[..., 47:53], axis=-1)
            if t0 is None
            else jnp.asarray(t0)
        )
        pool_capacity = (
            (y[..., 33] + y[..., 35] <= self.parameters[72] + 1.0e-6)
            & (y[..., 34] + y[..., 36] <= self.parameters[72] + 1.0e-6)
            & (y[..., 37] <= self.parameters[77] + 1.0e-6)
            & (y[..., 38] <= self.parameters[77] + 1.0e-6)
            & (free_troponin >= -1.0e-6)
        )
        return (
            parameter_valid
            & finite
            & positive_ions
            & gates
            & ryr_probability
            & nonnegative_pools
            & pool_capacity
        )


class ShortenCellStatus(IntFlag):
    SUCCESS = 0
    NONFINITE = 1
    SOLVER_FAILURE = 2
    INADMISSIBLE = 4
    TIME_MISALIGNMENT = 8
    INVALID_STEP = 16


class ShortenCellState(StrictModule):
    time_ms: Array
    values: Array

    def __init__(self, time_ms: ArrayLike, values: ArrayLike, /):
        time = jnp.asarray(time_ms)
        state = STATE_LAYOUT.require(values)
        if time.shape != ():
            raise ValueError("ShortenCellState time_ms must be scalar.")
        self.time_ms = time.astype(state.dtype)
        self.values = state


class _ShortenIntegrationSchedule(StrictModule, NonTrainableState):
    """Fixed solver lattice and stimulus protocol excluded from training."""

    time_grid_ms: Array
    protocol: ShortenPulseProtocol
    schedule_id: str = eqx.field(static=True)

    def __init__(
        self,
        time_grid_ms: ArrayLike,
        protocol: ShortenPulseProtocol,
        /,
    ):
        if not isinstance(protocol, ShortenPulseProtocol):
            raise TypeError("protocol must be a ShortenPulseProtocol.")
        grid = np.asarray(time_grid_ms)
        if grid.ndim != 1 or grid.size < 2 or not np.all(np.isfinite(grid)):
            raise ValueError("time_grid_ms must be a finite one-dimensional grid.")
        if not np.all(np.diff(grid) > 0.0):
            raise ValueError("time_grid_ms must be strictly increasing.")
        event_times = np.asarray(protocol.event_times_ms())
        interior = event_times[(event_times > grid[0]) & (event_times < grid[-1])]
        missing = tuple(
            float(event)
            for event in interior
            if not np.any(np.isclose(grid, event, rtol=0.0, atol=1.0e-12))
        )
        if missing:
            raise ValueError(
                "time_grid_ms must pin every stimulus start/end in its support; "
                f"missing={missing}."
            )
        time_grid = jnp.asarray(grid)
        self.time_grid_ms = time_grid
        self.protocol = protocol
        self.schedule_id = canonical_fingerprint(
            {
                "kind": "shorten-integration-schedule",
                "time_grid_ms": array_tree_fingerprint(time_grid),
                "protocol": protocol.protocol_id,
            }
        )


class ShortenIntegrationPlan(StrictModule):
    """Fixed event-aligned output lattice and stiff adaptive reaction policy."""

    model: ShortenFastTwitchModel
    schedule: _ShortenIntegrationSchedule
    relative_tolerance: float = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    initial_step_ms: float = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        model: ShortenFastTwitchModel,
        time_grid_ms: ArrayLike,
        /,
        *,
        protocol: ShortenPulseProtocol | None = None,
        relative_tolerance: float = 1.0e-6,
        absolute_tolerance: float = 1.0e-8,
        initial_step_ms: float = 1.0e-5,
        maximum_steps: int = 131072,
    ):
        if not isinstance(model, ShortenFastTwitchModel):
            raise TypeError("model must be a ShortenFastTwitchModel.")
        selected_protocol = (
            _DEFAULT_PULSE_PROTOCOL if protocol is None else protocol
        )
        schedule = _ShortenIntegrationSchedule(time_grid_ms, selected_protocol)
        scalars = (relative_tolerance, absolute_tolerance, initial_step_ms)
        if any(
            isinstance(value, bool)
            or not isfinite(float(value))
            or float(value) <= 0.0
            for value in scalars
        ):
            raise ValueError(
                "Integration tolerances and initial step must be positive."
            )
        if (
            not isinstance(maximum_steps, int)
            or isinstance(maximum_steps, bool)
            or maximum_steps <= 0
        ):
            raise ValueError("maximum_steps must be a positive integer.")
        self.model = model
        self.schedule = schedule
        self.relative_tolerance = float(relative_tolerance)
        self.absolute_tolerance = float(absolute_tolerance)
        self.initial_step_ms = float(initial_step_ms)
        self.maximum_steps = maximum_steps
        self.plan_id = canonical_fingerprint(
            {
                "kind": "shorten-fast-twitch-kvaerno5-event-aligned",
                "model": model.model_id,
                "schedule": schedule.schedule_id,
                "relative_tolerance": float(relative_tolerance),
                "absolute_tolerance": float(absolute_tolerance),
                "initial_step_ms": float(initial_step_ms),
                "maximum_steps": maximum_steps,
            }
        )

    @property
    def protocol(self) -> ShortenPulseProtocol:
        return self.schedule.protocol

    @property
    def time_grid_ms(self) -> Array:
        return self.schedule.time_grid_ms

    def prepare(self) -> PreparedShortenIntegrator:
        return PreparedShortenIntegrator(self)


class ShortenStepCandidate(StrictModule):
    previous: ShortenCellState
    proposed: ShortenCellState
    evaluation: ShortenFastTwitchEvaluation
    successful: Array
    status: Array
    solver_successful: Array
    solver_steps: Array
    plan_id: str = eqx.field(static=True)

    def commit(self) -> ShortenCellState:
        """Commit the complete candidate, or roll back time and all 56 states."""
        return ShortenCellState(
            jnp.where(self.successful, self.proposed.time_ms, self.previous.time_ms),
            jnp.where(self.successful, self.proposed.values, self.previous.values),
        )


class ShortenTrajectory(StrictModule):
    times_ms: Array
    states: Array
    successful: Array
    status: Array
    model_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class _ShortenDrift(StrictModule):
    model: ShortenFastTwitchModel
    protocol: ShortenPulseProtocol

    def __call__(self, time: Array, state: Array, args: object, /) -> Array:
        del args
        return self.model.rhs(time, state, protocol=self.protocol)


class PreparedShortenIntegrator(StrictModule):
    """Prepared Kvaerno5 route; exact gates remain available on the model."""

    plan: ShortenIntegrationPlan
    solver: dfx.Kvaerno5
    method_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, plan: ShortenIntegrationPlan, /):
        if not isinstance(plan, ShortenIntegrationPlan):
            raise TypeError("plan must be a ShortenIntegrationPlan.")
        self.plan = plan
        self.solver = dfx.Kvaerno5()
        self.method_id = "diffrax-kvaerno5-stiff-adaptive-source-complete"
        self.plan_id = plan.plan_id

    def initialize(self, *, dtype: object | None = None) -> ShortenCellState:
        return ShortenCellState(
            self.plan.time_grid_ms[0], self.plan.model.initialize(dtype=dtype)
        )

    def candidate(
        self,
        state: ShortenCellState,
        step_index: ArrayLike,
        /,
    ) -> ShortenStepCandidate:
        if not isinstance(state, ShortenCellState):
            raise TypeError("state must be a ShortenCellState.")
        index = jnp.asarray(step_index, dtype=jnp.int32)
        if index.shape != ():
            raise ValueError("step_index must be scalar.")
        valid_index = (index >= 0) & (
            index < self.plan.time_grid_ms.shape[0] - 1
        )
        index = jnp.clip(index, 0, self.plan.time_grid_ms.shape[0] - 2)
        start = self.plan.time_grid_ms[index].astype(state.values.dtype)
        end = self.plan.time_grid_ms[index + 1].astype(state.values.dtype)
        aligned = jnp.isclose(state.time_ms, start, rtol=0.0, atol=1.0e-10)
        problem = DifferentialProblem(
            _ShortenDrift(self.plan.model, self.plan.protocol),
            state.values,
            t0=start,
            t1=end,
            problem_id=f"shorten-cell-step:{self.plan_id}",
        )
        solution = solve_diffrax(
            problem,
            save_times=jnp.reshape(end, (1,)),
            solver=self.solver,
            dt0=self.plan.initial_step_ms,
            rtol=self.plan.relative_tolerance,
            atol=self.plan.absolute_tolerance,
            max_steps=self.plan.maximum_steps,
            throw=False,
            solver_configuration_id=self.method_id,
        )
        proposed_values = solution.states[-1]
        evaluation = self.plan.model.evaluate(
            end, proposed_values, protocol=self.plan.protocol
        )
        finite = jnp.all(jnp.isfinite(proposed_values)) & jnp.all(
            jnp.isfinite(evaluation.state_rate_per_ms)
        )
        solver_ok = jnp.asarray(solution.backend_successful) & jnp.all(solution.valid)
        admissible = jnp.all(evaluation.valid)
        successful = valid_index & aligned & solver_ok & finite & admissible
        status = (
            jnp.where(finite, 0, int(ShortenCellStatus.NONFINITE))
            | jnp.where(solver_ok, 0, int(ShortenCellStatus.SOLVER_FAILURE))
            | jnp.where(admissible, 0, int(ShortenCellStatus.INADMISSIBLE))
            | jnp.where(aligned, 0, int(ShortenCellStatus.TIME_MISALIGNMENT))
            | jnp.where(valid_index, 0, int(ShortenCellStatus.INVALID_STEP))
        ).astype(jnp.int32)
        return ShortenStepCandidate(
            state,
            ShortenCellState(end, proposed_values),
            evaluation,
            successful,
            status,
            solver_ok,
            jnp.asarray(solution.stats["num_steps"], dtype=jnp.int32),
            self.plan_id,
        )

    def integrate(self, initial: ShortenCellState | None = None) -> ShortenTrajectory:
        state = self.initialize() if initial is None else initial
        if not isinstance(state, ShortenCellState):
            raise TypeError("initial must be a ShortenCellState or None.")
        indices = jnp.arange(self.plan.time_grid_ms.shape[0] - 1, dtype=jnp.int32)

        def advance(current: ShortenCellState, index: Array):
            candidate = self.candidate(current, index)
            committed = candidate.commit()
            return committed, (
                committed.time_ms,
                committed.values,
                candidate.successful,
                candidate.status,
            )

        final, (times, values, successful, status) = jax.lax.scan(
            advance, state, indices
        )
        del final
        times = jnp.concatenate((state.time_ms[None], times), axis=0)
        states = jnp.concatenate((state.values[None, ...], values), axis=0)
        return ShortenTrajectory(
            times,
            states,
            successful,
            status,
            self.plan.model.model_id,
            self.plan_id,
        )


__all__ = [
    "PreparedShortenIntegrator",
    "ShortenCellState",
    "ShortenCellStatus",
    "ShortenFastTwitchEvaluation",
    "ShortenFastTwitchModel",
    "ShortenIntegrationPlan",
    "ShortenPulseProtocol",
    "ShortenStepCandidate",
    "ShortenTrajectory",
]
