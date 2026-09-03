#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Independently implemented ventricular cellular reaction models.

The implementations follow the published mathematical model families rather
than translated simulator source.  Currents are outward-positive and returned
as physical surface densities.  Model-specific state vectors are fixed and are
never padded to a cross-model union.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
from typing import ClassVar

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ._membrane_scaling import CardiacMembraneScaling
from ._reaction import (
    ArrayLike,
    CardiacReactionEvaluation,
    CardiacReactionParameterLayout,
    CardiacReactionStateLayout,
)
from ._reaction_ir import (
    compile_reaction_ir,
    CompiledReactionIR,
    PinnedReactionIR,
    ReactionBinaryOperator,
    ReactionIRBinary,
    ReactionIRInput,
    ReactionIROutput,
)


_GAS_CONSTANT_J_PER_KMOL_K = 8314.46261815324
_FARADAY_C_PER_MOL = 96485.33212


class VentricularCellPhenotype(Enum):
    """Typed transmural ventricular phenotype; strings are not admitted routes."""

    ENDOCARDIAL = "endocardial"
    MIDMYOCARDIAL = "midmyocardial"
    EPICARDIAL = "epicardial"


def _configured_model_id(
    family: str,
    phenotype: VentricularCellPhenotype,
    scaling: CardiacMembraneScaling,
    parameters: list[float],
) -> str:
    configuration = (
        f"{family}\0{phenotype.value}\0"
        f"{scaling.membrane_surface_to_volume_per_mm:.17g}\0"
        f"{scaling.membrane_capacitance_uF_per_mm2:.17g}\0"
        f"{tuple(parameters)!r}"
    )
    digest = sha256(configuration.encode("ascii")).hexdigest()[:16]
    return f"{family}:{phenotype.value}:{digest}"


def _sigmoid(x: Array) -> Array:
    return jnp.reciprocal(1.0 + jnp.exp(-x))


def _ghk_core(z: Array, concentration_in: Array, concentration_out: Array) -> Array:
    """Return z(ci exp(z)-co)/(exp(z)-1), including its analytic z=0 limit."""
    small = jnp.abs(z) < 1.0e-5
    denominator = jnp.where(small, jnp.ones_like(z), jnp.expm1(z))
    regular = z * (concentration_in * jnp.exp(z) - concentration_out) / denominator
    difference = concentration_in - concentration_out
    series = (
        difference
        + 0.5 * (concentration_in + concentration_out) * z
        + difference * z * z / 12.0
    )
    return jnp.where(small, series, regular)


def _exact_relaxation(
    value: Array,
    steady_state: Array,
    time_constant_ms: Array,
    dt_ms: Array,
) -> Array:
    return steady_state + (value - steady_state) * jnp.exp(-dt_ms / time_constant_ms)


def _ohmic_ir(program_name: str) -> CompiledReactionIR:
    conductance = ReactionIRInput(0)
    gate = ReactionIRInput(1)
    voltage = ReactionIRInput(2)
    reversal = ReactionIRInput(3)
    expression = ReactionIRBinary(
        ReactionBinaryOperator.MULTIPLY,
        ReactionIRBinary(ReactionBinaryOperator.MULTIPLY, conductance, gate),
        ReactionIRBinary(ReactionBinaryOperator.SUBTRACT, voltage, reversal),
    )
    return compile_reaction_ir(
        PinnedReactionIR(
            program_name=program_name,
            input_names=("conductance", "gate_product", "voltage_mV", "reversal_mV"),
            outputs=(ReactionIROutput("outward_current", expression),),
        )
    )


class _VentricularReactionBase:
    state_layout: CardiacReactionStateLayout
    parameter_layout: CardiacReactionParameterLayout
    default_parameters: Array
    current_names: tuple[str, ...]
    model_id: str
    scaling: CardiacMembraneScaling
    reaction_ir: CompiledReactionIR

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
        return jnp.all(jnp.isfinite(parameters) & (parameters >= 0.0), axis=-1)

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
        valid_dt = jnp.isfinite(dt) & (dt >= 0.0)
        gate_indices = jnp.asarray(self.state_layout.gate_indices)
        old_gates = resolved[..., gate_indices]
        updated_gates = _exact_relaxation(
            old_gates,
            evaluation.gate_steady_state,
            evaluation.gate_time_constant_ms,
            dt[..., None],
        )
        updated = resolved.at[..., gate_indices].set(updated_gates)
        valid = evaluation.valid & valid_dt
        return jnp.where(valid[..., None], updated, jnp.nan)

    def validate_state(
        self,
        state: ArrayLike,
        parameters: ArrayLike | None = None,
    ) -> None:
        array = np.asarray(state)
        if array.ndim == 0 or array.shape[-1] != self.state_layout.state_count:
            raise ValueError(
                "reaction state must have final axis size "
                f"{self.state_layout.state_count}, received shape {array.shape}."
            )
        if not np.all(np.isfinite(array)):
            raise ValueError("reaction state must be finite.")
        concentrations = array[..., self.state_layout.concentration_indices]
        if not np.all(concentrations > 0.0):
            raise ValueError("all ionic concentrations must be positive.")
        if not np.all(np.asarray(self.admissible(jnp.asarray(array), None))):
            raise ValueError("reaction state is outside the model's admissible domain.")
        if parameters is not None:
            parameter_array = np.asarray(parameters)
            if (
                parameter_array.ndim == 0
                or parameter_array.shape[-1] != self.parameter_layout.parameter_count
            ):
                raise ValueError(
                    "reaction parameter array has the wrong final axis size."
                )
            if not np.all(
                np.asarray(self._parameter_admissible(jnp.asarray(parameter_array)))
            ):
                raise ValueError("reaction parameters are outside their physical domain.")


_TP06_STATE_NAMES = (
    "voltage_mV",
    "xr1",
    "xr2",
    "xs",
    "m",
    "h",
    "j",
    "d",
    "f",
    "f2",
    "fCass",
    "s",
    "r",
    "calcium_i_mM",
    "calcium_sr_mM",
    "calcium_ss_mM",
    "ryanodine_available",
    "sodium_i_mM",
    "potassium_i_mM",
)
_TP06_GATE_NAMES = _TP06_STATE_NAMES[1:13]
_TP06_CONCENTRATIONS = (
    "calcium_i_mM",
    "calcium_sr_mM",
    "calcium_ss_mM",
    "sodium_i_mM",
    "potassium_i_mM",
)
_TP06_PARAMETER_NAMES = (
    "temperature_K",
    "sodium_o_mM",
    "potassium_o_mM",
    "calcium_o_mM",
    "g_Na_mS_per_uF",
    "g_CaL_L_per_F_ms",
    "g_to_mS_per_uF",
    "g_Kr_mS_per_uF",
    "g_Ks_mS_per_uF",
    "g_K1_mS_per_uF",
    "g_bNa_mS_per_uF",
    "g_bCa_mS_per_uF",
    "g_pK_mS_per_uF",
    "g_pCa_mS_per_uF",
    "p_NaK_pA_per_pF",
    "k_NaCa_pA_per_pF",
    "cell_capacitance_uF",
    "cytosol_volume_uL",
    "sr_volume_uL",
    "ss_volume_uL",
    "calcium_up_mM_per_ms",
    "calcium_release_per_ms",
    "calcium_leak_per_ms",
    "calcium_transfer_per_ms",
)
_TP06_DEFAULT_PARAMETERS = (
    310.0,
    140.0,
    5.4,
    2.0,
    14.838,
    0.000175,
    0.294,
    0.153,
    0.392,
    5.405,
    0.00029,
    0.000592,
    0.0146,
    0.1238,
    2.724,
    1000.0,
    0.185,
    0.016404,
    0.001094,
    0.00005468,
    0.006375,
    0.102,
    0.00036,
    0.0038,
)


@dataclass(frozen=True)
class TenTusscherPanfilov2006Model(_VentricularReactionBase):
    """Nineteen-state human ventricular ten Tusscher--Panfilov 2006 model."""

    phenotype: VentricularCellPhenotype = VentricularCellPhenotype.EPICARDIAL
    scaling: CardiacMembraneScaling = field(default_factory=CardiacMembraneScaling)
    model_id: str = field(init=False)
    default_parameters: Array = field(init=False, repr=False, compare=False)
    reaction_ir: CompiledReactionIR = field(init=False, repr=False, compare=False)

    state_layout: ClassVar[CardiacReactionStateLayout] = CardiacReactionStateLayout(
        _TP06_STATE_NAMES,
        (
            "mV",
            "1",
            "1",
            "1",
            "1",
            "1",
            "1",
            "1",
            "1",
            "1",
            "1",
            "1",
            "1",
            "mM",
            "mM",
            "mM",
            "1",
            "mM",
            "mM",
        ),
        _TP06_GATE_NAMES,
        _TP06_CONCENTRATIONS,
    )
    parameter_layout: ClassVar[CardiacReactionParameterLayout] = (
        CardiacReactionParameterLayout(
            _TP06_PARAMETER_NAMES,
            (
                "K",
                "mM",
                "mM",
                "mM",
                "mS/uF",
                "L/(F ms)",
                "mS/uF",
                "mS/uF",
                "mS/uF",
                "mS/uF",
                "mS/uF",
                "mS/uF",
                "mS/uF",
                "mS/uF",
                "pA/pF",
                "pA/pF",
                "uF",
                "uL",
                "uL",
                "uL",
                "mM/ms",
                "1/ms",
                "1/ms",
                "1/ms",
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
        "I_NaCa",
        "I_NaK",
        "I_pCa",
        "I_pK",
        "I_bNa",
        "I_bCa",
    )

    def __post_init__(self) -> None:
        if not isinstance(self.phenotype, VentricularCellPhenotype):
            raise TypeError("phenotype must be VentricularCellPhenotype.")
        if not isinstance(self.scaling, CardiacMembraneScaling):
            raise TypeError("scaling must be CardiacMembraneScaling.")
        values = list(_TP06_DEFAULT_PARAMETERS)
        if self.phenotype is VentricularCellPhenotype.ENDOCARDIAL:
            values[6] = 0.073
        elif self.phenotype is VentricularCellPhenotype.MIDMYOCARDIAL:
            values[6] = 0.294
            values[8] = 0.098
        object.__setattr__(self, "default_parameters", jnp.asarray(values))
        object.__setattr__(
            self,
            "model_id",
            _configured_model_id(
                "ten-tusscher-panfilov-2006",
                self.phenotype,
                self.scaling,
                values,
            ),
        )
        object.__setattr__(self, "reaction_ir", _ohmic_ir("tp06-ohmic-current-v1"))

    def initialize(
        self,
        batch_shape: tuple[int, ...] = (),
        *,
        dtype: object | None = None,
    ) -> Array:
        resolved_dtype = jnp.asarray(0.0).dtype if dtype is None else dtype
        initial = jnp.asarray(
            (
                -86.2,
                0.0,
                1.0,
                0.0,
                0.0,
                0.75,
                0.75,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.0,
                0.00007,
                1.3,
                0.00007,
                1.0,
                7.67,
                138.3,
            ),
            dtype=resolved_dtype,
        )
        return jnp.broadcast_to(initial, tuple(batch_shape) + initial.shape)

    def _parameter_admissible(self, parameters: Array) -> Array:
        positive = jnp.concatenate((parameters[..., :4], parameters[..., 16:]), axis=-1)
        nonnegative = parameters[..., 4:16]
        return (
            jnp.all(jnp.isfinite(parameters), axis=-1)
            & jnp.all(positive > 0.0, axis=-1)
            & jnp.all(nonnegative >= 0.0, axis=-1)
        )

    def admissible(self, state: Array, parameters: Array | None = None) -> Array:
        resolved = self.state_layout.require_shape(state)
        p = self._parameters(parameters, resolved.dtype)
        finite = jnp.all(jnp.isfinite(resolved), axis=-1) & jnp.all(
            jnp.isfinite(p), axis=-1
        )
        concentrations = resolved[
            ..., jnp.asarray(self.state_layout.concentration_indices)
        ]
        gates = resolved[..., 1:13]
        voltage = resolved[..., 0]
        ryanodine = resolved[..., 16]
        return (
            finite
            & self._parameter_admissible(p)
            & jnp.all(concentrations > 0.0, axis=-1)
            & jnp.all((gates >= 0.0) & (gates <= 1.0), axis=-1)
            & (ryanodine >= 0.0)
            & (ryanodine <= 1.0)
            & (voltage > -200.0)
            & (voltage < 150.0)
        )

    def evaluate(
        self,
        state: Array,
        parameters: Array | None = None,
        *,
        stimulus_current_uA_per_mm2: ArrayLike = 0.0,
    ) -> CardiacReactionEvaluation:
        y = self.state_layout.require_shape(state)
        p = self._parameters(parameters, y.dtype)
        pi = self.parameter_layout.index
        v = y[..., 0]
        xr1, xr2, xs = y[..., 1], y[..., 2], y[..., 3]
        m, h, j = y[..., 4], y[..., 5], y[..., 6]
        d, f, f2, fcass = y[..., 7], y[..., 8], y[..., 9], y[..., 10]
        s, r = y[..., 11], y[..., 12]
        cai, casr, cass = y[..., 13], y[..., 14], y[..., 15]
        rprime, nai, ki = y[..., 16], y[..., 17], y[..., 18]

        temperature = p[..., pi("temperature_K")]
        nao = p[..., pi("sodium_o_mM")]
        ko = p[..., pi("potassium_o_mM")]
        cao = p[..., pi("calcium_o_mM")]
        rtf = _GAS_CONSTANT_J_PER_KMOL_K * temperature / _FARADAY_C_PER_MOL
        vfrt = v / rtf
        ena = rtf * jnp.log(nao / nai)
        ek = rtf * jnp.log(ko / ki)
        eks = rtf * jnp.log((ko + 0.03 * nao) / (ki + 0.03 * nai))
        eca = 0.5 * rtf * jnp.log(cao / cai)

        i_na = p[..., pi("g_Na_mS_per_uF")] * m**3 * h * j * (v - ena)
        z_cal = 2.0 * (v - 15.0) / rtf
        i_cal = (
            2.0
            * p[..., pi("g_CaL_L_per_F_ms")]
            * _FARADAY_C_PER_MOL
            * d
            * f
            * f2
            * fcass
            * _ghk_core(z_cal, 0.25 * cass, cao)
        )
        i_to = p[..., pi("g_to_mS_per_uF")] * r * s * (v - ek)
        i_kr = p[..., pi("g_Kr_mS_per_uF")] * jnp.sqrt(ko / 5.4) * xr1 * xr2 * (v - ek)
        i_ks = p[..., pi("g_Ks_mS_per_uF")] * xs**2 * (v - eks)
        alpha_k1 = 0.1 / (1.0 + jnp.exp(0.06 * (v - ek - 200.0)))
        beta_k1 = (
            3.0 * jnp.exp(0.0002 * (v - ek + 100.0)) + jnp.exp(0.1 * (v - ek - 10.0))
        ) / (1.0 + jnp.exp(-0.5 * (v - ek)))
        xk1_inf = alpha_k1 / (alpha_k1 + beta_k1)
        i_k1 = p[..., pi("g_K1_mS_per_uF")] * xk1_inf * (v - ek)
        exchanger_numerator = (
            jnp.exp(0.35 * vfrt) * nai**3 * cao
            - jnp.exp(-0.65 * vfrt) * nao**3 * cai * 2.5
        )
        exchanger_denominator = (
            (87.5**3 + nao**3) * (1.38 + cao) * (1.0 + 0.1 * jnp.exp(-0.65 * vfrt))
        )
        i_naca = (
            p[..., pi("k_NaCa_pA_per_pF")] * exchanger_numerator / exchanger_denominator
        )
        i_nak = (
            p[..., pi("p_NaK_pA_per_pF")]
            * ko
            / (ko + 1.0)
            * nai
            / (nai + 40.0)
            / (1.0 + 0.1245 * jnp.exp(-0.1 * vfrt) + 0.0353 * jnp.exp(-vfrt))
        )
        i_pca = p[..., pi("g_pCa_mS_per_uF")] * cai / (0.0005 + cai)
        i_pk = (
            p[..., pi("g_pK_mS_per_uF")] * (v - ek) / (1.0 + jnp.exp((25.0 - v) / 5.98))
        )
        i_bna = p[..., pi("g_bNa_mS_per_uF")] * (v - ena)
        i_bca = p[..., pi("g_bCa_mS_per_uF")] * (v - eca)
        normalized_currents = jnp.stack(
            (
                i_na,
                i_cal,
                i_to,
                i_kr,
                i_ks,
                i_k1,
                i_naca,
                i_nak,
                i_pca,
                i_pk,
                i_bna,
                i_bca,
            ),
            axis=-1,
        )
        current_density = normalized_currents * self.membrane_capacitance_uF_per_mm2
        total_current = jnp.sum(current_density, axis=-1)
        stimulus = jnp.asarray(stimulus_current_uA_per_mm2, dtype=y.dtype)
        dv = -(total_current + stimulus) / self.membrane_capacitance_uF_per_mm2

        xr1_inf = _sigmoid((v + 26.0) / 7.0)
        tau_xr1 = 450.0 / (1.0 + jnp.exp((-45.0 - v) / 10.0))
        tau_xr1 *= 6.0 / (1.0 + jnp.exp((v + 30.0) / 11.5))
        xr2_inf = _sigmoid(-(v + 88.0) / 24.0)
        tau_xr2 = 3.0 / (1.0 + jnp.exp((-60.0 - v) / 20.0))
        tau_xr2 *= 1.12 / (1.0 + jnp.exp((v - 60.0) / 20.0))
        xs_inf = _sigmoid((v + 5.0) / 14.0)
        tau_xs = 1400.0 / jnp.sqrt(1.0 + jnp.exp((5.0 - v) / 6.0))
        tau_xs /= 1.0 + jnp.exp((v - 35.0) / 15.0)
        tau_xs += 80.0
        m_inf = _sigmoid((v + 56.86) / 9.03) ** 2
        tau_m = 1.0 / (1.0 + jnp.exp((-60.0 - v) / 5.0))
        tau_m *= 0.1 / (1.0 + jnp.exp((v + 35.0) / 5.0)) + 0.1 / (
            1.0 + jnp.exp((v - 50.0) / 200.0)
        )
        h_inf = _sigmoid(-(v + 71.55) / 7.43) ** 2
        alpha_h = jnp.where(v >= -40.0, 0.0, 0.057 * jnp.exp(-(v + 80.0) / 6.8))
        beta_h = jnp.where(
            v >= -40.0,
            0.77 / (0.13 * (1.0 + jnp.exp(-(v + 10.66) / 11.1))),
            2.7 * jnp.exp(0.079 * v) + 310000.0 * jnp.exp(0.3485 * v),
        )
        tau_h = 1.0 / (alpha_h + beta_h)
        alpha_j = jnp.where(
            v >= -40.0,
            0.0,
            (-25428.0 * jnp.exp(0.2444 * v) - 6.948e-6 * jnp.exp(-0.04391 * v))
            * (v + 37.78)
            / (1.0 + jnp.exp(0.311 * (v + 79.23))),
        )
        beta_j = jnp.where(
            v >= -40.0,
            0.6 * jnp.exp(0.057 * v) / (1.0 + jnp.exp(-0.1 * (v + 32.0))),
            0.02424 * jnp.exp(-0.01052 * v) / (1.0 + jnp.exp(-0.1378 * (v + 40.14))),
        )
        tau_j = 1.0 / (alpha_j + beta_j)
        d_inf = _sigmoid((v + 8.0) / 7.5)
        tau_d = (1.4 / (1.0 + jnp.exp((-35.0 - v) / 13.0)) + 0.25) * 1.4 / (
            1.0 + jnp.exp((v + 5.0) / 5.0)
        ) + 1.0 / (1.0 + jnp.exp((50.0 - v) / 20.0))
        f_inf = _sigmoid(-(v + 20.0) / 7.0)
        tau_f = (
            1102.5 * jnp.exp(-((v + 27.0) ** 2) / 225.0)
            + 200.0 / (1.0 + jnp.exp((13.0 - v) / 10.0))
            + 180.0 / (1.0 + jnp.exp((v + 30.0) / 10.0))
            + 20.0
        )
        f2_inf = 0.67 / (1.0 + jnp.exp((v + 35.0) / 7.0)) + 0.33
        tau_f2 = (
            562.0 * jnp.exp(-((v + 27.0) ** 2) / 240.0)
            + 31.0 / (1.0 + jnp.exp((25.0 - v) / 10.0))
            + 80.0 / (1.0 + jnp.exp((v + 30.0) / 10.0))
        )
        cass_ratio2 = (cass / 0.05) ** 2
        fcass_inf = 0.6 / (1.0 + cass_ratio2) + 0.4
        tau_fcass = 80.0 / (1.0 + cass_ratio2) + 2.0
        if self.phenotype is VentricularCellPhenotype.ENDOCARDIAL:
            s_inf = _sigmoid(-(v + 28.0) / 5.0)
            tau_s = 1000.0 * jnp.exp(-((v + 67.0) ** 2) / 1000.0) + 8.0
        else:
            s_inf = _sigmoid(-(v + 20.0) / 5.0)
            tau_s = (
                85.0 * jnp.exp(-((v + 45.0) ** 2) / 320.0)
                + 5.0 / (1.0 + jnp.exp((v - 20.0) / 5.0))
                + 3.0
            )
        r_inf = _sigmoid((v - 20.0) / 6.0)
        tau_r = 9.5 * jnp.exp(-((v + 40.0) ** 2) / 1800.0) + 0.8
        gate_inf = jnp.stack(
            (
                xr1_inf,
                xr2_inf,
                xs_inf,
                m_inf,
                h_inf,
                h_inf,
                d_inf,
                f_inf,
                f2_inf,
                fcass_inf,
                s_inf,
                r_inf,
            ),
            axis=-1,
        )
        gate_tau = jnp.stack(
            (
                tau_xr1,
                tau_xr2,
                tau_xs,
                tau_m,
                tau_h,
                tau_j,
                tau_d,
                tau_f,
                tau_f2,
                tau_fcass,
                tau_s,
                tau_r,
            ),
            axis=-1,
        )
        gate_rates = (gate_inf - y[..., 1:13]) / gate_tau

        kcasr = 2.5 - 1.5 / (1.0 + (0.45 / casr) ** 2)
        k1 = 0.15 / kcasr
        k2 = 0.045 * kcasr
        open_probability = k1 * cass**2 * rprime / (0.06 + k1 * cass**2)
        i_rel = p[..., pi("calcium_release_per_ms")] * open_probability * (casr - cass)
        i_up = p[..., pi("calcium_up_mM_per_ms")] / (1.0 + (0.00025 / cai) ** 2)
        i_leak = p[..., pi("calcium_leak_per_ms")] * (casr - cai)
        i_xfer = p[..., pi("calcium_transfer_per_ms")] * (cass - cai)
        d_rprime = 0.005 * (1.0 - rprime) - k2 * cass * rprime
        buffer_i = 1.0 / (1.0 + 0.2 * 0.001 / (cai + 0.001) ** 2)
        buffer_sr = 1.0 / (1.0 + 10.0 * 0.3 / (casr + 0.3) ** 2)
        buffer_ss = 1.0 / (1.0 + 0.4 * 0.00025 / (cass + 0.00025) ** 2)
        capacitance = p[..., pi("cell_capacitance_uF")]
        vc = p[..., pi("cytosol_volume_uL")]
        vsr = p[..., pi("sr_volume_uL")]
        vss = p[..., pi("ss_volume_uL")]
        calcium_membrane_norm = i_bca + i_pca - 2.0 * i_naca
        d_cai = buffer_i * (
            (i_leak - i_up) * vsr / vc
            + i_xfer
            - calcium_membrane_norm * capacitance / (2.0 * vc * _FARADAY_C_PER_MOL)
        )
        d_casr = buffer_sr * (i_up - i_leak - i_rel)
        d_cass = buffer_ss * (
            -i_cal * capacitance / (2.0 * vss * _FARADAY_C_PER_MOL)
            + i_rel * vsr / vss
            - i_xfer * vc / vss
        )
        d_nai = (
            -(i_na + i_bna + 3.0 * i_nak + 3.0 * i_naca)
            * capacitance
            / (vc * _FARADAY_C_PER_MOL)
        )
        d_ki = (
            -(i_k1 + i_to + i_kr + i_ks - 2.0 * i_nak + i_pk)
            * capacitance
            / (vc * _FARADAY_C_PER_MOL)
        )
        state_rate = jnp.concatenate(
            (
                dv[..., None],
                gate_rates,
                d_cai[..., None],
                d_casr[..., None],
                d_cass[..., None],
                d_rprime[..., None],
                d_nai[..., None],
                d_ki[..., None],
            ),
            axis=-1,
        )
        valid = self.admissible(y, p)
        nan = jnp.asarray(jnp.nan, dtype=y.dtype)
        state_rate = jnp.where(valid[..., None], state_rate, nan)
        gate_inf = jnp.where(valid[..., None], gate_inf, nan)
        gate_tau = jnp.where(valid[..., None], gate_tau, nan)
        current_density = jnp.where(valid[..., None], current_density, nan)
        total_current = jnp.where(valid, total_current, nan)
        calcium_membrane_current = (
            i_cal + calcium_membrane_norm
        ) * self.membrane_capacitance_uF_per_mm2
        calcium_membrane_current = jnp.where(valid, calcium_membrane_current, nan)
        sr_flux = jnp.where(valid, i_rel + i_leak - i_up, nan)
        charge_residual = (
            self.membrane_capacitance_uF_per_mm2 * state_rate[..., 0]
            + total_current
            + stimulus
        )
        return CardiacReactionEvaluation(
            state_rate=state_rate,
            gate_steady_state=gate_inf,
            gate_time_constant_ms=gate_tau,
            current_density_uA_per_mm2=current_density,
            total_outward_current_uA_per_mm2=total_current,
            calcium_cytosol_mM=jnp.where(valid, cai, nan),
            calcium_cytosol_rate_mM_per_ms=jnp.where(valid, d_cai, nan),
            calcium_sr_flux_mM_per_ms=sr_flux,
            calcium_membrane_current_uA_per_mm2=calcium_membrane_current,
            charge_balance_residual_uA_per_mm2=charge_residual,
            valid=valid,
            current_names=self.current_names,
            model_id=self.model_id,
        )


_ORD_STATE_NAMES = (
    "voltage_mV",
    "calmodulin_kinase_trapped",
    "sodium_i_mM",
    "sodium_ss_mM",
    "potassium_i_mM",
    "potassium_ss_mM",
    "calcium_i_mM",
    "calcium_ss_mM",
    "calcium_nsr_mM",
    "calcium_jsr_mM",
    "m",
    "h_fast",
    "h_slow",
    "j",
    "h_slow_phosphorylated",
    "j_phosphorylated",
    "m_late",
    "h_late",
    "h_late_phosphorylated",
    "a_to",
    "i_to_fast",
    "i_to_slow",
    "a_to_phosphorylated",
    "i_to_fast_phosphorylated",
    "i_to_slow_phosphorylated",
    "d",
    "f_fast",
    "f_slow",
    "f_ca_fast",
    "f_ca_slow",
    "j_ca",
    "n_ca",
    "f_fast_phosphorylated",
    "f_ca_fast_phosphorylated",
    "xr_fast",
    "xr_slow",
    "xs1",
    "xs2",
    "xk1",
    "release_nonphosphorylated_mM_per_ms",
    "release_phosphorylated_mM_per_ms",
)
_ORD_CONCENTRATIONS = _ORD_STATE_NAMES[2:10]
_ORD_GATE_NAMES = _ORD_STATE_NAMES[10:]
_ORD_PARAMETER_NAMES = (
    "temperature_K",
    "sodium_o_mM",
    "potassium_o_mM",
    "calcium_o_mM",
    "g_Na_mS_per_uF",
    "g_NaL_mS_per_uF",
    "g_to_mS_per_uF",
    "p_CaL_L_per_F_ms",
    "g_Kr_mS_per_uF",
    "g_Ks_mS_per_uF",
    "g_K1_mS_per_uF",
    "g_NaCa_pA_per_pF",
    "g_NaK_pA_per_pF",
    "g_Kb_mS_per_uF",
    "p_Nab_L_per_F_ms",
    "p_Cab_L_per_F_ms",
    "g_pCa_mS_per_uF",
    "cell_capacitance_uF",
    "cytosol_volume_uL",
    "ss_volume_uL",
    "nsr_volume_uL",
    "jsr_volume_uL",
    "calcium_up_mM_per_ms",
    "calcium_release_scale_per_ms",
)
_ORD_DEFAULT_PARAMETERS = (
    310.0,
    140.0,
    5.4,
    1.8,
    75.0,
    0.0075,
    0.02,
    0.0001,
    0.046,
    0.0034,
    0.1908,
    0.0008,
    30.0,
    0.003,
    3.75e-10,
    2.5e-8,
    0.0005,
    0.1534,
    0.02584,
    0.00076,
    0.002098,
    0.000182,
    0.004375,
    1.0,
)


@dataclass(frozen=True)
class ORdVentricularModel(_VentricularReactionBase):
    """Forty-one-state O'Hara--Rudy dynamic human ventricular model family."""

    phenotype: VentricularCellPhenotype = VentricularCellPhenotype.ENDOCARDIAL
    scaling: CardiacMembraneScaling = field(default_factory=CardiacMembraneScaling)
    model_id: str = field(init=False)
    default_parameters: Array = field(init=False, repr=False, compare=False)
    reaction_ir: CompiledReactionIR = field(init=False, repr=False, compare=False)

    state_layout: ClassVar[CardiacReactionStateLayout] = CardiacReactionStateLayout(
        _ORD_STATE_NAMES,
        (
            "mV",
            "1",
            "mM",
            "mM",
            "mM",
            "mM",
            "mM",
            "mM",
            "mM",
            "mM",
        )
        + ("1",) * 29
        + ("mM/ms", "mM/ms"),
        _ORD_GATE_NAMES,
        _ORD_CONCENTRATIONS,
    )
    parameter_layout: ClassVar[CardiacReactionParameterLayout] = (
        CardiacReactionParameterLayout(
            _ORD_PARAMETER_NAMES,
            (
                "K",
                "mM",
                "mM",
                "mM",
                "mS/uF",
                "mS/uF",
                "mS/uF",
                "L/(F ms)",
                "mS/uF",
                "mS/uF",
                "mS/uF",
                "pA/pF",
                "pA/pF",
                "mS/uF",
                "L/(F ms)",
                "L/(F ms)",
                "mS/uF",
                "uF",
                "uL",
                "uL",
                "uL",
                "uL",
                "mM/ms",
                "1/ms",
            ),
        )
    )
    current_names: ClassVar[tuple[str, ...]] = (
        "I_Na",
        "I_NaL",
        "I_to",
        "I_CaL",
        "I_CaNa",
        "I_CaK",
        "I_Kr",
        "I_Ks",
        "I_K1",
        "I_NaCa_i",
        "I_NaCa_ss",
        "I_NaK",
        "I_Kb",
        "I_Nab",
        "I_Cab",
        "I_pCa",
    )

    def __post_init__(self) -> None:
        if not isinstance(self.phenotype, VentricularCellPhenotype):
            raise TypeError("phenotype must be VentricularCellPhenotype.")
        if not isinstance(self.scaling, CardiacMembraneScaling):
            raise TypeError("scaling must be CardiacMembraneScaling.")
        values = list(_ORD_DEFAULT_PARAMETERS)
        if self.phenotype is VentricularCellPhenotype.EPICARDIAL:
            values[5] *= 0.6
            values[6] *= 4.0
            values[7] *= 1.2
            values[8] *= 1.3
            values[9] *= 1.4
            values[10] *= 1.2
            values[11] *= 1.1
            values[12] *= 0.9
            values[13] *= 0.6
            values[22] *= 1.3
        elif self.phenotype is VentricularCellPhenotype.MIDMYOCARDIAL:
            values[5] *= 1.0
            values[6] *= 4.0
            values[7] *= 2.5
            values[8] *= 0.8
            values[9] *= 1.0
            values[10] *= 1.3
            values[11] *= 1.4
            values[12] *= 0.7
        object.__setattr__(self, "default_parameters", jnp.asarray(values))
        object.__setattr__(
            self,
            "model_id",
            _configured_model_id(
                "ohara-rudy-dynamic-2011",
                self.phenotype,
                self.scaling,
                values,
            ),
        )
        object.__setattr__(self, "reaction_ir", _ohmic_ir("ord-ohmic-current-v1"))

    def initialize(
        self,
        batch_shape: tuple[int, ...] = (),
        *,
        dtype: object | None = None,
    ) -> Array:
        resolved_dtype = jnp.asarray(0.0).dtype if dtype is None else dtype
        initial = jnp.asarray(
            (
                -87.0,
                0.0,
                7.0,
                7.0,
                145.0,
                145.0,
                0.0001,
                0.0001,
                1.2,
                1.2,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.0,
                1.0,
                1.0,
                0.0,
                1.0,
                1.0,
                0.0,
                1.0,
                1.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
            ),
            dtype=resolved_dtype,
        )
        return jnp.broadcast_to(initial, tuple(batch_shape) + initial.shape)

    def _parameter_admissible(self, parameters: Array) -> Array:
        positive = jnp.concatenate((parameters[..., :4], parameters[..., 17:]), axis=-1)
        nonnegative = parameters[..., 4:17]
        return (
            jnp.all(jnp.isfinite(parameters), axis=-1)
            & jnp.all(positive > 0.0, axis=-1)
            & jnp.all(nonnegative >= 0.0, axis=-1)
        )

    def admissible(self, state: Array, parameters: Array | None = None) -> Array:
        y = self.state_layout.require_shape(state)
        p = self._parameters(parameters, y.dtype)
        concentrations = y[..., 2:10]
        bounded_gates = y[..., 10:39]
        return (
            jnp.all(jnp.isfinite(y), axis=-1)
            & self._parameter_admissible(p)
            & jnp.all(concentrations > 0.0, axis=-1)
            & (y[..., 1] >= 0.0)
            & (y[..., 1] <= 1.0)
            & jnp.all((bounded_gates >= 0.0) & (bounded_gates <= 1.0), axis=-1)
            & (y[..., 0] > -200.0)
            & (y[..., 0] < 150.0)
        )

    def evaluate(
        self,
        state: Array,
        parameters: Array | None = None,
        *,
        stimulus_current_uA_per_mm2: ArrayLike = 0.0,
    ) -> CardiacReactionEvaluation:
        y = self.state_layout.require_shape(state)
        p = self._parameters(parameters, y.dtype)
        pi = self.parameter_layout.index
        v, camkt = y[..., 0], y[..., 1]
        nai, nass, ki, kss = y[..., 2], y[..., 3], y[..., 4], y[..., 5]
        cai, cass, cansr, cajsr = y[..., 6], y[..., 7], y[..., 8], y[..., 9]
        (
            m,
            hf,
            hs,
            j,
            hsp,
            jp,
            ml,
            hl,
            hlp,
            a,
            i_f,
            i_s,
            ap,
            i_fp,
            i_sp,
            d,
            ff,
            fs,
            fcaf,
            fcas,
            jca,
            nca,
            ffp,
            fcafp,
            xrf,
            xrs,
            xs1,
            xs2,
            xk1,
            jrelnp,
            jrelp,
        ) = (y[..., index] for index in range(10, 41))

        temperature = p[..., pi("temperature_K")]
        nao = p[..., pi("sodium_o_mM")]
        ko = p[..., pi("potassium_o_mM")]
        cao = p[..., pi("calcium_o_mM")]
        rtf = _GAS_CONSTANT_J_PER_KMOL_K * temperature / _FARADAY_C_PER_MOL
        vfrt = v / rtf
        ena = rtf * jnp.log(nao / nai)
        ek = rtf * jnp.log(ko / ki)
        eks = rtf * jnp.log((ko + 0.01833 * nao) / (ki + 0.01833 * nai))

        camkb = 0.05 * (1.0 - camkt) / (1.0 + 0.0015 / cass)
        camka = camkb + camkt
        f_phosphorylated = 1.0 / (1.0 + 0.15 / camka)
        h = 0.99 * hf + 0.01 * hs
        hp = 0.99 * hf + 0.01 * hsp
        i_na = (
            p[..., pi("g_Na_mS_per_uF")]
            * (v - ena)
            * m**3
            * ((1.0 - f_phosphorylated) * h * j + f_phosphorylated * hp * jp)
        )
        i_nal = (
            p[..., pi("g_NaL_mS_per_uF")]
            * (v - ena)
            * ml
            * ((1.0 - f_phosphorylated) * hl + f_phosphorylated * hlp)
        )
        ai_f = 1.0 / (1.0 + jnp.exp((v - 213.6) / 151.2))
        i_gate = ai_f * i_f + (1.0 - ai_f) * i_s
        ip_gate = ai_f * i_fp + (1.0 - ai_f) * i_sp
        i_to = (
            p[..., pi("g_to_mS_per_uF")]
            * (v - ek)
            * ((1.0 - f_phosphorylated) * a * i_gate + f_phosphorylated * ap * ip_gate)
        )
        aff = 0.6
        afcaf = 0.3 + 0.6 / (1.0 + jnp.exp((v - 10.0) / 10.0))
        f_voltage = aff * ff + (1.0 - aff) * fs
        f_calcium = afcaf * fcaf + (1.0 - afcaf) * fcas
        f_voltage_p = aff * ffp + (1.0 - aff) * fs
        f_calcium_p = afcaf * fcafp + (1.0 - afcaf) * fcas
        gate_np = f_voltage * (1.0 - nca) + jca * f_calcium * nca
        gate_p = f_voltage_p * (1.0 - nca) + jca * f_calcium_p * nca
        calcium_gate = d * (
            (1.0 - f_phosphorylated) * gate_np + f_phosphorylated * gate_p
        )
        z_ca = 2.0 * vfrt
        phi_cal = 2.0 * _FARADAY_C_PER_MOL * _ghk_core(z_ca, cass, 0.341 * cao)
        phi_cana = _FARADAY_C_PER_MOL * _ghk_core(vfrt, 0.75 * nass, 0.75 * nao)
        phi_cak = _FARADAY_C_PER_MOL * _ghk_core(vfrt, 0.75 * kss, 0.75 * ko)
        pcal = p[..., pi("p_CaL_L_per_F_ms")]
        effective_pcal = pcal * (1.0 + 0.1 * f_phosphorylated)
        i_cal = effective_pcal * phi_cal * calcium_gate
        i_cana = 0.00125 * effective_pcal * phi_cana * calcium_gate
        i_cak = 0.0003574 * effective_pcal * phi_cak * calcium_gate
        axrf = 1.0 / (1.0 + jnp.exp((v + 54.81) / 38.21))
        xr = axrf * xrf + (1.0 - axrf) * xrs
        rkr = 1.0 / (1.0 + jnp.exp((v + 55.0) / 75.0))
        rkr /= 1.0 + jnp.exp((v - 10.0) / 30.0)
        i_kr = p[..., pi("g_Kr_mS_per_uF")] * jnp.sqrt(ko / 5.4) * xr * rkr * (v - ek)
        ks_ca = 1.0 + 0.6 / (1.0 + (3.8e-5 / cai) ** 1.4)
        i_ks = p[..., pi("g_Ks_mS_per_uF")] * ks_ca * xs1 * xs2 * (v - eks)
        rk1 = 1.0 / (1.0 + jnp.exp((v + 105.8 - 2.6 * ko) / 9.493))
        i_k1 = p[..., pi("g_K1_mS_per_uF")] * jnp.sqrt(ko) * rk1 * xk1 * (v - ek)

        def exchanger(local_na: Array, local_ca: Array) -> Array:
            h_na = jnp.exp(0.5224 * vfrt)
            h_ca = jnp.exp(0.1670 * vfrt)
            h1 = 1.0 + local_na / 88.12 * (1.0 + h_na)
            h2 = local_na * h_na / (88.12 * h1)
            h3 = 1.0 / h1
            h4 = 1.0 + local_na / 15.0 * (1.0 + local_na / 5.0)
            h5 = local_na**2 / (h4 * 15.0 * 5.0)
            h6 = 1.0 / h4
            h7 = 1.0 + nao / 88.12 * (1.0 + 1.0 / h_na)
            h8 = nao / (88.12 * h_na * h7)
            h9 = 1.0 / h7
            h10 = 12.5 + 1.0 + nao / 15.0 * (1.0 + nao / 5.0)
            h11 = nao**2 / (h10 * 15.0 * 5.0)
            h12 = 1.0 / h10
            k1 = h12 * cao * 1.5e6
            k2 = 5.0e3
            k3p = h9 * 6.0e4
            k3pp = h8 * 5.0e3
            k3 = k3p + k3pp
            k4p = h3 * 6.0e4 / h_ca
            k4pp = h2 * 5.0e3
            k4 = k4p + k4pp
            k5 = 5.0e3
            k6 = h6 * local_ca * 1.5e6
            k7 = h5 * h2 * 6.0e4
            k8 = h8 * h11 * 6.0e4
            x1_ncx = k2 * k4 * (k7 + k6) + k5 * k7 * (k2 + k3)
            x2_ncx = k1 * k7 * (k4 + k5) + k4 * k6 * (k1 + k8)
            x3_ncx = k1 * k3 * (k7 + k6) + k8 * k6 * (k2 + k3)
            x4_ncx = k2 * k8 * (k4 + k5) + k3 * k5 * (k1 + k8)
            occupancy_sum = x1_ncx + x2_ncx + x3_ncx + x4_ncx
            e1_ncx = x1_ncx / occupancy_sum
            e2_ncx = x2_ncx / occupancy_sum
            e3_ncx = x3_ncx / occupancy_sum
            e4_ncx = x4_ncx / occupancy_sum
            sodium_flux = (
                3.0 * (e4_ncx * k7 - e1_ncx * k8) + e3_ncx * k4pp - e2_ncx * k3pp
            )
            calcium_flux = e2_ncx * k2 - e1_ncx * k1
            allosteric = 1.0 / (1.0 + (0.00015 / local_ca) ** 2)
            return (
                p[..., pi("g_NaCa_pA_per_pF")]
                * allosteric
                * (sodium_flux + 2.0 * calcium_flux)
            )

        i_naca_i = 0.8 * exchanger(nai, cai)
        i_naca_ss = 0.2 * exchanger(nass, cass)
        knai = 9.073 * jnp.exp(-0.155 * vfrt / 3.0)
        knao = 27.78 * jnp.exp(1.155 * vfrt / 3.0)
        denominator_i = (1.0 + nai / knai) ** 3 + (1.0 + ki / 0.5) ** 2 - 1.0
        denominator_o = (1.0 + nao / knao) ** 3 + (1.0 + ko / 0.3582) ** 2 - 1.0
        pump_phosphorylation = 4.2 / (1.0 + 1.0e-7 / 1.698e-7 + nai / 224.0 + ki / 292.0)
        alpha1 = 949.5 * (nai / knai) ** 3 / denominator_i
        beta1 = 182.4 * 0.05
        alpha2 = 687.2
        beta2 = 39.4 * (nao / knao) ** 3 / denominator_o
        alpha3 = 1899.0 * (ko / 0.3582) ** 2 / denominator_o
        beta3 = 79300.0 * pump_phosphorylation * 1.0e-7 / (1.0 + 9.8 / 1.698e-7)
        alpha4 = 639.0 * (9.8 / 1.698e-7) / (1.0 + 9.8 / 1.698e-7)
        beta4 = 40.0 * (ki / 0.5) ** 2 / denominator_i
        x1 = (
            alpha4 * alpha1 * alpha2
            + beta2 * beta4 * beta3
            + alpha2 * beta4 * beta3
            + beta3 * alpha1 * alpha2
        )
        x2 = (
            beta2 * beta1 * beta4
            + alpha1 * alpha2 * alpha3
            + alpha3 * beta1 * beta4
            + alpha2 * alpha3 * beta4
        )
        x3 = (
            alpha2 * alpha3 * alpha4
            + beta3 * beta2 * beta1
            + beta2 * beta1 * alpha4
            + alpha3 * alpha4 * beta1
        )
        x4 = (
            beta4 * beta3 * beta2
            + alpha3 * alpha4 * alpha1
            + beta2 * alpha4 * alpha1
            + beta3 * beta2 * alpha1
        )
        pump_sum = x1 + x2 + x3 + x4
        e1, e2, e3, e4 = x1 / pump_sum, x2 / pump_sum, x3 / pump_sum, x4 / pump_sum
        sodium_flux = 3.0 * (e1 * alpha3 - e2 * beta3)
        potassium_flux = 2.0 * (e4 * beta1 - e3 * alpha1)
        i_nak = p[..., pi("g_NaK_pA_per_pF")] * (sodium_flux + potassium_flux)
        xkb = 1.0 / (1.0 + jnp.exp(-(v - 14.48) / 18.34))
        i_kb = p[..., pi("g_Kb_mS_per_uF")] * xkb * (v - ek)
        i_nab = (
            p[..., pi("p_Nab_L_per_F_ms")]
            * _FARADAY_C_PER_MOL
            * _ghk_core(vfrt, nai, nao)
        )
        i_cab = (
            p[..., pi("p_Cab_L_per_F_ms")]
            * 2.0
            * _FARADAY_C_PER_MOL
            * _ghk_core(z_ca, cai, 0.341 * cao)
        )
        i_pca = p[..., pi("g_pCa_mS_per_uF")] * cai / (0.0005 + cai)
        normalized_currents = jnp.stack(
            (
                i_na,
                i_nal,
                i_to,
                i_cal,
                i_cana,
                i_cak,
                i_kr,
                i_ks,
                i_k1,
                i_naca_i,
                i_naca_ss,
                i_nak,
                i_kb,
                i_nab,
                i_cab,
                i_pca,
            ),
            axis=-1,
        )
        current_density = normalized_currents * self.membrane_capacitance_uF_per_mm2
        total_current = jnp.sum(current_density, axis=-1)
        stimulus = jnp.asarray(stimulus_current_uA_per_mm2, dtype=y.dtype)
        dv = -(total_current + stimulus) / self.membrane_capacitance_uF_per_mm2

        mss = _sigmoid((v + 39.57) / 9.871)
        tm = 1.0 / (
            6.765 * jnp.exp((v + 11.64) / 34.77) + 8.552 * jnp.exp(-(v + 77.42) / 5.955)
        )
        hss = _sigmoid(-(v + 82.9) / 6.086)
        thf = 1.0 / (
            1.432e-5 * jnp.exp(-(v + 1.196) / 6.285)
            + 6.149 * jnp.exp((v + 0.5096) / 20.27)
        )
        ths = 1.0 / (
            0.009794 * jnp.exp(-(v + 17.95) / 28.05)
            + 0.3343 * jnp.exp((v + 5.73) / 56.66)
        )
        jss = hss
        tj = 2.038 + 1.0 / (
            0.02136 * jnp.exp(-(v + 100.6) / 8.281)
            + 0.3052 * jnp.exp((v + 0.9941) / 38.45)
        )
        hssp = _sigmoid(-(v + 89.1) / 6.086)
        mlss = _sigmoid((v + 42.85) / 5.264)
        hlss = _sigmoid(-(v + 87.61) / 7.488)
        hlssp = _sigmoid(-(v + 93.81) / 7.488)
        ass = _sigmoid((v - 14.34) / 14.82)
        ta = 1.0515 / (
            1.0 / (1.2089 * (1.0 + jnp.exp(-(v - 18.4099) / 29.3814)))
            + 3.5 / (1.0 + jnp.exp((v + 100.0) / 29.3814))
        )
        iss = _sigmoid(-(v + 43.94) / 5.711)
        tif = 4.562 + 1.0 / (
            0.3933 * jnp.exp(-(v + 100.0) / 100.0) + 0.08004 * jnp.exp((v + 50.0) / 16.59)
        )
        tis = 23.62 + 1.0 / (
            0.001416 * jnp.exp(-(v + 96.52) / 59.05)
            + 1.78e-8 * jnp.exp((v + 114.1) / 8.079)
        )
        if self.phenotype is VentricularCellPhenotype.EPICARDIAL:
            delta_epi = 1.0 - 0.95 / (1.0 + jnp.exp((v + 70.0) / 5.0))
            tif = tif * delta_epi
            tis = tis * delta_epi
        develop = 1.354 + 0.0001 / (
            jnp.exp((v - 167.4) / 15.89) + jnp.exp(-(v - 12.23) / 0.2154)
        )
        recover = 1.0 - 0.5 / (1.0 + jnp.exp((v + 70.0) / 20.0))
        dss = _sigmoid((v + 3.94) / 4.23)
        td = 0.6 + 1.0 / (jnp.exp(-0.05 * (v + 6.0)) + jnp.exp(0.09 * (v + 14.0)))
        fss = _sigmoid(-(v + 19.58) / 3.696)
        tff = 7.0 + 1.0 / (
            0.0045 * jnp.exp(-(v + 20.0) / 10.0) + 0.0045 * jnp.exp((v + 20.0) / 10.0)
        )
        tfs = 1000.0 + 1.0 / (
            0.000035 * jnp.exp(-(v + 5.0) / 4.0) + 0.000035 * jnp.exp((v + 5.0) / 6.0)
        )
        tfcaf = 7.0 + 1.0 / (
            0.04 * jnp.exp(-(v - 4.0) / 7.0) + 0.04 * jnp.exp((v - 4.0) / 7.0)
        )
        tfcas = 100.0 + 1.0 / (0.00012 * jnp.exp(-v / 3.0) + 0.00012 * jnp.exp(v / 7.0))
        k2n = 1000.0
        anca = jca / (k2n + jca * (1.0 + 0.002 / cass) ** 4)
        nca_forward = anca * k2n
        nca_backward = jca
        nca_inf = nca_forward / (nca_forward + nca_backward)
        tnca = 1.0 / (nca_forward + nca_backward)
        xrss = _sigmoid((v + 8.337) / 6.789)
        txrf = 12.98 + 1.0 / (
            0.3652 * jnp.exp((v - 31.66) / 3.869)
            + 4.123e-5 * jnp.exp(-(v - 47.78) / 20.38)
        )
        txrs = 1.865 + 1.0 / (
            0.06629 * jnp.exp((v - 34.7) / 7.355)
            + 1.128e-5 * jnp.exp(-(v - 29.74) / 25.94)
        )
        xs1ss = _sigmoid((v + 11.6) / 8.932)
        txs1 = 817.3 + 1.0 / (
            0.0002326 * jnp.exp((v + 48.28) / 17.8)
            + 0.001292 * jnp.exp(-(v + 210.0) / 230.0)
        )
        txs2 = 1.0 / (
            0.01 * jnp.exp((v - 50.0) / 20.0) + 0.0193 * jnp.exp(-(v + 66.54) / 31.0)
        )
        xk1ss = _sigmoid((v + 2.5538 * ko + 144.59) / (1.5692 * ko + 3.8115))
        txk1 = 122.2 / (jnp.exp(-(v + 127.2) / 20.36) + jnp.exp((v + 236.8) / 69.33))

        release_time_scale = 4.75
        if self.phenotype is VentricularCellPhenotype.MIDMYOCARDIAL:
            release_time_scale *= 1.7
        release_denominator = 1.0 + (1.5 / cajsr) ** 8
        jrel_inf = (
            -0.5
            * release_time_scale
            * p[..., pi("calcium_release_scale_per_ms")]
            * i_cal
            / release_denominator
        )
        tau_rel = jnp.maximum(
            release_time_scale / (1.0 + 0.0123 / cajsr),
            0.001,
        )
        phosphorylated_release_time_scale = 1.25 * release_time_scale
        jrelp_inf = (
            -0.5
            * phosphorylated_release_time_scale
            * p[..., pi("calcium_release_scale_per_ms")]
            * i_cal
            / release_denominator
        )
        tau_relp = jnp.maximum(
            phosphorylated_release_time_scale / (1.0 + 0.0123 / cajsr),
            0.001,
        )
        gate_inf = jnp.stack(
            (
                mss,
                hss,
                hss,
                jss,
                hssp,
                jss,
                mlss,
                hlss,
                hlssp,
                ass,
                iss,
                iss,
                ass,
                iss,
                iss,
                dss,
                fss,
                fss,
                fss,
                fss,
                fss,
                nca_inf,
                fss,
                fss,
                xrss,
                xrss,
                xs1ss,
                xs1ss,
                xk1ss,
                jrel_inf,
                jrelp_inf,
            ),
            axis=-1,
        )
        gate_tau = jnp.stack(
            (
                tm,
                thf,
                ths,
                tj,
                3.0 * ths,
                1.46 * tj,
                tm,
                jnp.full_like(v, 200.0),
                jnp.full_like(v, 600.0),
                ta,
                tif,
                tis,
                ta,
                tif * develop * recover,
                tis * develop * recover,
                td,
                tff,
                tfs,
                tfcaf,
                tfcas,
                jnp.full_like(v, 75.0),
                tnca,
                2.5 * tff,
                2.5 * tfcaf,
                txrf,
                txrs,
                txs1,
                txs2,
                txk1,
                tau_rel,
                tau_relp,
            ),
            axis=-1,
        )
        gate_rates = (gate_inf - y[..., 10:41]) / gate_tau
        d_camkt = 0.05 * camkb * (camkb + camkt) - 0.00068 * camkt

        jdiff_na = (nass - nai) / 2.0
        jdiff_k = (kss - ki) / 2.0
        jdiff_ca = (cass - cai) / 0.2
        jtr = (cansr - cajsr) / 100.0
        jup_np = p[..., pi("calcium_up_mM_per_ms")] * cai / (cai + 0.00092)
        jup_p = 2.75 * p[..., pi("calcium_up_mM_per_ms")] * cai / (cai + 0.00075)
        jleak = 0.0039375 * cansr / 15.0
        jup = (1.0 - f_phosphorylated) * jup_np + f_phosphorylated * jup_p - jleak
        jrel = (1.0 - f_phosphorylated) * jrelnp + f_phosphorylated * jrelp
        capacitance = p[..., pi("cell_capacitance_uF")]
        vmyo = p[..., pi("cytosol_volume_uL")]
        vss = p[..., pi("ss_volume_uL")]
        vnsr = p[..., pi("nsr_volume_uL")]
        vjsr = p[..., pi("jsr_volume_uL")]
        d_nai = (
            -(i_na + i_nal + 3.0 * i_naca_i + 3.0 * i_nak)
            * capacitance
            / (_FARADAY_C_PER_MOL * vmyo)
            + jdiff_na * vss / vmyo
        )
        d_nass = (
            -(i_cana + 3.0 * i_naca_ss) * capacitance / (_FARADAY_C_PER_MOL * vss)
            - jdiff_na
        )
        d_ki = (
            -(i_to + i_kr + i_ks + i_k1 + i_kb - 2.0 * i_nak)
            * capacitance
            / (_FARADAY_C_PER_MOL * vmyo)
            + jdiff_k * vss / vmyo
        )
        d_kss = -i_cak * capacitance / (_FARADAY_C_PER_MOL * vss) - jdiff_k
        beta_cai = 1.0 / (
            1.0
            + 0.05 * 0.00238 / (0.00238 + cai) ** 2
            + 0.07 * 0.0005 / (0.0005 + cai) ** 2
        )
        beta_cass = 1.0 / (
            1.0
            + 0.047 * 0.00087 / (0.00087 + cass) ** 2
            + 1.124 * 0.0087 / (0.0087 + cass) ** 2
        )
        beta_cajsr = 1.0 / (1.0 + 10.0 * 0.8 / (0.8 + cajsr) ** 2)
        calcium_membrane_i = i_cab + i_pca - 2.0 * i_naca_i
        calcium_membrane_ss = i_cal - 2.0 * i_naca_ss
        d_cai = beta_cai * (
            -calcium_membrane_i * capacitance / (2.0 * _FARADAY_C_PER_MOL * vmyo)
            - jup * vnsr / vmyo
            + jdiff_ca * vss / vmyo
        )
        d_cass = beta_cass * (
            -calcium_membrane_ss * capacitance / (2.0 * _FARADAY_C_PER_MOL * vss)
            + jrel * vjsr / vss
            - jdiff_ca
        )
        d_cansr = jup - jtr * vjsr / vnsr
        d_cajsr = beta_cajsr * (jtr - jrel)
        state_rate = jnp.concatenate(
            (
                dv[..., None],
                d_camkt[..., None],
                d_nai[..., None],
                d_nass[..., None],
                d_ki[..., None],
                d_kss[..., None],
                d_cai[..., None],
                d_cass[..., None],
                d_cansr[..., None],
                d_cajsr[..., None],
                gate_rates,
            ),
            axis=-1,
        )
        valid = self.admissible(y, p)
        nan = jnp.asarray(jnp.nan, dtype=y.dtype)
        state_rate = jnp.where(valid[..., None], state_rate, nan)
        gate_inf = jnp.where(valid[..., None], gate_inf, nan)
        gate_tau = jnp.where(valid[..., None], gate_tau, nan)
        current_density = jnp.where(valid[..., None], current_density, nan)
        total_current = jnp.where(valid, total_current, nan)
        calcium_membrane_current = (
            i_cal + i_cab + i_pca - 2.0 * (i_naca_i + i_naca_ss)
        ) * self.membrane_capacitance_uF_per_mm2
        calcium_membrane_current = jnp.where(valid, calcium_membrane_current, nan)
        sr_flux = jnp.where(valid, jrel - jup, nan)
        charge_residual = (
            self.membrane_capacitance_uF_per_mm2 * state_rate[..., 0]
            + total_current
            + stimulus
        )
        return CardiacReactionEvaluation(
            state_rate=state_rate,
            gate_steady_state=gate_inf,
            gate_time_constant_ms=gate_tau,
            current_density_uA_per_mm2=current_density,
            total_outward_current_uA_per_mm2=total_current,
            calcium_cytosol_mM=jnp.where(valid, cai, nan),
            calcium_cytosol_rate_mM_per_ms=jnp.where(valid, d_cai, nan),
            calcium_sr_flux_mM_per_ms=sr_flux,
            calcium_membrane_current_uA_per_mm2=calcium_membrane_current,
            charge_balance_residual_uA_per_mm2=charge_residual,
            valid=valid,
            current_names=self.current_names,
            model_id=self.model_id,
        )


__all__ = [
    "ORdVentricularModel",
    "TenTusscherPanfilov2006Model",
    "VentricularCellPhenotype",
]
