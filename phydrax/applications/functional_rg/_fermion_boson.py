#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._regulators import FunctionalRGStatus
from ._wetterich import _volume_factor


FermionBosonRepresentation = Literal["gross-neveu", "yukawa", "mixed"]


class FiniteTemperatureThresholds(StrictModule):
    boson: Array
    fermion: Array
    mixed: Array
    boson_tail_indicator: Array
    fermion_tail_indicator: Array
    finite: Array
    admissible: Array
    status: Array
    plan_id: str = eqx.field(static=True)


class MatsubaraThresholdPlan(StrictModule, NonTrainableState):
    """Explicit finite bosonic/fermionic Matsubara sums with a hard mode budget."""

    __hash__ = object.__hash__

    boson_indices: Array
    fermion_indices: Array
    maximum_mode: int = eqx.field(static=True)
    maximum_terms: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, maximum_mode: int, /, *, maximum_terms: int = 4096):
        mode = int(maximum_mode)
        capacity = int(maximum_terms)
        term_count = (2 * mode + 1) + 2 * mode
        if mode < 1 or capacity <= 0 or term_count > capacity:
            raise ValueError("Matsubara mode cutoff exceeds the fixed term budget.")
        boson = np.arange(-mode, mode + 1, dtype=np.int32)
        fermion = np.arange(-mode, mode, dtype=np.int32)
        self.boson_indices = jnp.asarray(boson)
        self.fermion_indices = jnp.asarray(fermion)
        self.maximum_mode = mode
        self.maximum_terms = capacity
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-matsubara-threshold-plan",
                "maximum_mode": mode,
                "maximum_terms": capacity,
                "boson_indices": array_tree_fingerprint(boson),
                "fermion_indices": array_tree_fingerprint(fermion),
            }
        )

    def evaluate(
        self,
        temperature_over_scale: ArrayLike,
        boson_mass_squared: ArrayLike = 0.0,
        fermion_mass_squared: ArrayLike = 0.0,
        /,
    ) -> FiniteTemperatureThresholds:
        temperature = jnp.asarray(temperature_over_scale)
        boson_mass = jnp.asarray(boson_mass_squared)
        fermion_mass = jnp.asarray(fermion_mass_squared)
        temperature, boson_mass, fermion_mass = jnp.broadcast_arrays(
            temperature, boson_mass, fermion_mass
        )
        safe_temperature = jnp.where(temperature > 0.0, temperature, 1.0)
        trailing = (1,) * temperature.ndim
        boson_frequency = (
            2.0
            * jnp.pi
            * self.boson_indices.reshape(self.boson_indices.shape + trailing)
            * safe_temperature
        )
        fermion_frequency = (
            (2.0 * self.fermion_indices + 1.0).reshape(
                self.fermion_indices.shape + trailing
            )
            * jnp.pi
            * safe_temperature
        )
        boson_gap = 1.0 + boson_mass
        fermion_gap = 1.0 + fermion_mass
        boson_terms = 1.0 / (boson_frequency**2 + boson_gap) ** 2
        fermion_terms = 1.0 / (fermion_frequency**2 + fermion_gap) ** 2
        boson_sum = 4.0 * safe_temperature * jnp.sum(boson_terms, axis=0)
        fermion_sum = 4.0 * safe_temperature * jnp.sum(fermion_terms, axis=0)
        zero_temperature_boson = boson_gap ** (-1.5)
        zero_temperature_fermion = fermion_gap ** (-1.5)
        boson = jnp.where(temperature == 0.0, zero_temperature_boson, boson_sum)
        fermion = jnp.where(temperature == 0.0, zero_temperature_fermion, fermion_sum)
        mixed = jnp.sqrt(jnp.maximum(boson * fermion, 0.0))
        boson_tail = jnp.where(
            temperature == 0.0,
            0.0,
            8.0 * safe_temperature * boson_terms[0] * (self.maximum_mode + 1),
        )
        fermion_tail = jnp.where(
            temperature == 0.0,
            0.0,
            8.0
            * safe_temperature
            * jnp.maximum(fermion_terms[0], fermion_terms[-1])
            * (self.maximum_mode + 0.5),
        )
        finite = (
            jnp.isfinite(temperature)
            & jnp.isfinite(boson)
            & jnp.isfinite(fermion)
            & jnp.isfinite(boson_tail)
            & jnp.isfinite(fermion_tail)
        )
        admissible = (
            finite & (temperature >= 0.0) & (boson_gap > 0.0) & (fermion_gap > 0.0)
        )
        status = jnp.where(
            admissible,
            int(FunctionalRGStatus.SUCCESS),
            jnp.where(
                finite,
                int(FunctionalRGStatus.POLE_ENCOUNTERED),
                int(FunctionalRGStatus.NONFINITE),
            ),
        ).astype(jnp.int32)
        return FiniteTemperatureThresholds(
            boson,
            fermion,
            mixed,
            boson_tail,
            fermion_tail,
            finite,
            admissible,
            status,
            self.plan_id,
        )


class FermionBosonTruncationState(StrictModule):
    scalar_mass_squared: Array
    scalar_quartic: Array
    yukawa_squared: Array
    four_fermion: Array

    def __init__(
        self,
        scalar_mass_squared: ArrayLike,
        scalar_quartic: ArrayLike,
        yukawa_squared: ArrayLike,
        four_fermion: ArrayLike,
        /,
    ):
        values = tuple(
            jnp.asarray(value).reshape(())
            for value in (
                scalar_mass_squared,
                scalar_quartic,
                yukawa_squared,
                four_fermion,
            )
        )
        self.scalar_mass_squared = values[0]
        self.scalar_quartic = values[1]
        self.yukawa_squared = values[2]
        self.four_fermion = values[3]

    def as_array(self, /) -> Array:
        return jnp.stack(
            (
                self.scalar_mass_squared,
                self.scalar_quartic,
                self.yukawa_squared,
                self.four_fermion,
            )
        )


class FermionBosonFlowEvaluation(StrictModule):
    beta: Array
    scalar_anomalous_dimension: Array
    fermion_anomalous_dimension: Array
    thresholds: FiniteTemperatureThresholds
    finite: Array
    admissible: Array
    status: Array
    plan_id: str = eqx.field(static=True)


class FermionBosonTruncationIdentityEvidence(StrictModule):
    inactive_beta_residual: Array
    bosonization_residual: Array
    finite: Array
    satisfied: Array
    plan_id: str = eqx.field(static=True)


class GrossNeveuYukawaFlowPlan(StrictModule, NonTrainableState):
    """Finite Gross--Neveu/Yukawa truncation with explicit thermal decoupling."""

    __hash__ = object.__hash__

    matsubara: MatsubaraThresholdPlan
    fermion_flavors: int = eqx.field(static=True)
    scalar_components: int = eqx.field(static=True)
    dimension: float = eqx.field(static=True)
    representation: FermionBosonRepresentation = eqx.field(static=True)
    volume_factor: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        fermion_flavors: int,
        scalar_components: int,
        dimension: float,
        matsubara: MatsubaraThresholdPlan,
        /,
        *,
        representation: FermionBosonRepresentation = "yukawa",
    ):
        flavors = int(fermion_flavors)
        scalars = int(scalar_components)
        dimension_ = float(dimension)
        if not isinstance(matsubara, MatsubaraThresholdPlan):
            raise TypeError("matsubara must be MatsubaraThresholdPlan.")
        if (
            flavors <= 0
            or scalars <= 0
            or not 2.0 < dimension_ <= 4.0
            or representation not in ("gross-neveu", "yukawa", "mixed")
        ):
            raise ValueError("Gross--Neveu/Yukawa truncation data are invalid.")
        self.matsubara = matsubara
        self.fermion_flavors = flavors
        self.scalar_components = scalars
        self.dimension = dimension_
        self.representation = representation
        self.volume_factor = _volume_factor(dimension_)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "gross-neveu-yukawa-finite-truncation",
                "fermion_flavors": flavors,
                "scalar_components": scalars,
                "dimension": dimension_,
                "matsubara": matsubara.plan_id,
                "representation": representation,
                "coupling_normalization": "dimensionless-derivative-expansion",
            }
        )

    def evaluate(
        self,
        state: FermionBosonTruncationState,
        temperature_over_scale: ArrayLike = 0.0,
        /,
    ) -> FermionBosonFlowEvaluation:
        if not isinstance(state, FermionBosonTruncationState):
            raise TypeError("state must be FermionBosonTruncationState.")
        thresholds = self.matsubara.evaluate(
            temperature_over_scale, state.scalar_mass_squared, 0.0
        )
        mass = state.scalar_mass_squared
        quartic = state.scalar_quartic
        yukawa = state.yukawa_squared
        four_fermion = state.four_fermion
        vd = self.volume_factor
        eta_scalar = 8.0 * vd * self.fermion_flavors * yukawa * thresholds.fermion
        eta_fermion = (
            8.0 * vd / self.dimension * self.scalar_components * yukawa * thresholds.boson
        )
        beta_mass = (-2.0 + eta_scalar) * mass + 4.0 * vd * (
            (self.scalar_components + 2.0) * quartic * thresholds.boson
            - 4.0 * self.fermion_flavors * yukawa * thresholds.fermion
        )
        beta_quartic = (self.dimension - 4.0 + 2.0 * eta_scalar) * quartic + 4.0 * vd * (
            (self.scalar_components + 8.0) * quartic**2 * thresholds.boson
            - 8.0 * self.fermion_flavors * yukawa**2 * thresholds.fermion
        )
        beta_yukawa = (
            self.dimension - 4.0 + eta_scalar + 2.0 * eta_fermion
        ) * yukawa + 8.0 * vd * (
            self.fermion_flavors + self.scalar_components
        ) * yukawa**2 * thresholds.mixed
        beta_four = (
            self.dimension - 2.0 + 2.0 * eta_fermion
        ) * four_fermion - 4.0 * vd * (
            2.0 * self.fermion_flavors - 1.0
        ) * four_fermion**2 * thresholds.fermion
        if self.representation == "gross-neveu":
            beta = jnp.stack(
                (
                    jnp.zeros_like(beta_mass),
                    jnp.zeros_like(beta_mass),
                    jnp.zeros_like(beta_mass),
                    beta_four,
                )
            )
            eta_scalar = jnp.zeros_like(eta_scalar)
        elif self.representation == "yukawa":
            beta = jnp.stack(
                (
                    beta_mass,
                    beta_quartic,
                    beta_yukawa,
                    jnp.zeros_like(beta_four),
                )
            )
        else:
            exchange = 2.0 * vd * yukawa**2 * thresholds.mixed
            beta = jnp.stack((beta_mass, beta_quartic, beta_yukawa, beta_four + exchange))
        finite = (
            thresholds.finite
            & jnp.all(jnp.isfinite(state.as_array()))
            & jnp.all(jnp.isfinite(beta))
        )
        admissible = thresholds.admissible & finite
        status = jnp.where(
            admissible,
            int(FunctionalRGStatus.SUCCESS),
            jnp.where(
                finite,
                int(FunctionalRGStatus.POLE_ENCOUNTERED),
                int(FunctionalRGStatus.NONFINITE),
            ),
        ).astype(jnp.int32)
        return FermionBosonFlowEvaluation(
            beta,
            eta_scalar,
            eta_fermion,
            thresholds,
            finite,
            admissible,
            status,
            self.plan_id,
        )

    def truncation_identity(
        self,
        state: FermionBosonTruncationState,
        temperature_over_scale: ArrayLike = 0.0,
        /,
    ) -> FermionBosonTruncationIdentityEvidence:
        evaluation = self.evaluate(state, temperature_over_scale)
        if self.representation == "gross-neveu":
            inactive = jnp.max(jnp.abs(evaluation.beta[:3]))
        elif self.representation == "yukawa":
            inactive = jnp.abs(evaluation.beta[3])
        else:
            inactive = jnp.asarray(0.0, dtype=evaluation.beta.dtype)
        if self.representation == "mixed":
            safe_mass = jnp.where(
                jnp.abs(state.scalar_mass_squared) > 1.0e-15,
                state.scalar_mass_squared,
                1.0,
            )
            bosonization = state.four_fermion - state.yukawa_squared / safe_mass
            bosonization = jnp.where(
                jnp.abs(state.scalar_mass_squared) > 1.0e-15,
                bosonization,
                jnp.inf,
            )
        else:
            bosonization = jnp.asarray(0.0, dtype=evaluation.beta.dtype)
        finite = evaluation.finite & jnp.isfinite(inactive) & jnp.isfinite(bosonization)
        tolerance = 64.0 * jnp.finfo(evaluation.beta.dtype).eps
        satisfied = (
            finite
            & (inactive <= tolerance)
            & ((self.representation != "mixed") | (jnp.abs(bosonization) <= tolerance))
        )
        return FermionBosonTruncationIdentityEvidence(
            inactive,
            bosonization,
            finite,
            satisfied,
            self.plan_id,
        )


__all__ = [
    "FermionBosonFlowEvaluation",
    "FermionBosonRepresentation",
    "FermionBosonTruncationIdentityEvidence",
    "FermionBosonTruncationState",
    "FiniteTemperatureThresholds",
    "GrossNeveuYukawaFlowPlan",
    "MatsubaraThresholdPlan",
]
