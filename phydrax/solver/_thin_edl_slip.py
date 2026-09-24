#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntFlag

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._admissibility import (
    AdmissibilityHeader,
    AdmissibilityReason,
)
from .._differentiation import (
    branch_policy_contract,
    BranchDifferentiationPolicy,
    DerivativeContract,
    DerivativeSurface,
)
from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.finite_volume._mac_boundary import MACBoundaryProvider
from ..equations._chemical_thermodynamics import UNIVERSAL_GAS_CONSTANT
from ..equations._electrochemistry import FARADAY_CONSTANT


class ThinEDLReason(IntFlag):
    DEBYE_RATIO_EXCEEDED = 1 << 8
    DUKHIN_NUMBER_EXCEEDED = 1 << 9
    BULK_NOT_ELECTRONEUTRAL = 1 << 10
    INVALID_TANGENTIAL_FIELD = 1 << 11


# Derivatives of the executed algorithm with model and regime decisions frozen.
_DERIVATIVE_CONTRACT = branch_policy_contract(
    BranchDifferentiationPolicy.FROZEN_DECISION,
    surfaces=(DerivativeSurface.PRIMAL_STATE, DerivativeSurface.PHYSICAL_PARAMETER),
)


class ThinEDLSlipEvaluation(StrictModule):
    slip_velocity: Array
    debye_length: Array
    debye_ratio: Array
    dukhin_number: Array
    electroneutrality_defect: Array
    header: AdmissibilityHeader
    derivative_contract: DerivativeContract
    volumetric_force_permitted: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def boundary_provider(self, /) -> MACBoundaryProvider:
        if not bool(np.asarray(self.header.globally_eligible)):
            raise ValueError("Thin-EDL slip cannot bind an inadmissible evaluation.")
        return MACBoundaryProvider(self.slip_velocity)


class ThinEDLElectroosmoticSlipPlan(StrictModule, NonTrainableState):
    """DC Helmholtz-Smoluchowski slip with explicit thin-layer admission."""

    permittivity: float = eqx.field(static=True)
    dynamic_viscosity: float = eqx.field(static=True)
    zeta_potential: float = eqx.field(static=True)
    characteristic_length: float = eqx.field(static=True)
    surface_conductivity: float = eqx.field(static=True)
    bulk_conductivity: float = eqx.field(static=True)
    maximum_debye_ratio: float = eqx.field(static=True)
    maximum_dukhin_number: float = eqx.field(static=True)
    electroneutrality_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        permittivity: float,
        dynamic_viscosity: float,
        zeta_potential: float,
        characteristic_length: float,
        surface_conductivity: float = 0.0,
        bulk_conductivity: float,
        maximum_debye_ratio: float = 0.05,
        maximum_dukhin_number: float = 0.1,
        electroneutrality_tolerance: float = 1.0e-8,
    ) -> None:
        values = tuple(
            float(value)
            for value in (
                permittivity,
                dynamic_viscosity,
                zeta_potential,
                characteristic_length,
                surface_conductivity,
                bulk_conductivity,
                maximum_debye_ratio,
                maximum_dukhin_number,
                electroneutrality_tolerance,
            )
        )
        if (
            any(not np.isfinite(value) for value in values)
            or values[0] <= 0.0
            or values[1] <= 0.0
            or values[3] <= 0.0
            or values[4] < 0.0
            or values[5] <= 0.0
            or values[6] <= 0.0
            or values[7] < 0.0
            or values[8] < 0.0
        ):
            raise ValueError("Thin-EDL physical parameters are invalid.")
        (
            self.permittivity,
            self.dynamic_viscosity,
            self.zeta_potential,
            self.characteristic_length,
            self.surface_conductivity,
            self.bulk_conductivity,
            self.maximum_debye_ratio,
            self.maximum_dukhin_number,
            self.electroneutrality_tolerance,
        ) = values
        self.plan_id = canonical_fingerprint(
            {
                "kind": "thin-edl-electroosmotic-slip",
                "parameters": values,
                "forcing": "dc-nonreactive",
                "volumetric_force": "forbidden",
            }
        )

    def evaluate(
        self,
        concentrations: ArrayLike,
        valences: ArrayLike,
        temperature: ArrayLike,
        electric_field: ArrayLike,
        outward_normal: ArrayLike,
        /,
    ) -> ThinEDLSlipEvaluation:
        concentration = jnp.asarray(concentrations)
        charge_number = jnp.asarray(valences, dtype=concentration.dtype)
        temperature_ = jnp.asarray(temperature, dtype=concentration.dtype)
        electric = jnp.asarray(electric_field, dtype=concentration.dtype)
        normal = jnp.asarray(outward_normal, dtype=concentration.dtype)
        if (
            concentration.ndim != 1
            or charge_number.shape != concentration.shape
            or temperature_.shape != ()
            or electric.ndim != 1
            or normal.shape != electric.shape
        ):
            raise ValueError(
                "Thin-EDL concentrations, field, or normal are incompatible."
            )
        ionic_strength_factor = jnp.sum(charge_number**2 * concentration)
        debye = jnp.sqrt(
            self.permittivity
            * UNIVERSAL_GAS_CONSTANT
            * temperature_
            / jnp.maximum(
                FARADAY_CONSTANT**2 * ionic_strength_factor,
                jnp.finfo(concentration.dtype).tiny,
            )
        )
        debye_ratio = debye / self.characteristic_length
        dukhin = self.surface_conductivity / (
            self.bulk_conductivity * self.characteristic_length
        )
        charge_defect = jnp.sum(charge_number * concentration)
        charge_scale = jnp.maximum(jnp.sum(jnp.abs(charge_number * concentration)), 1.0)
        electroneutral = (
            jnp.abs(charge_defect) <= self.electroneutrality_tolerance * charge_scale
        )
        normal_norm = jnp.sqrt(jnp.sum(normal**2))
        unit_normal = normal / jnp.maximum(normal_norm, jnp.finfo(normal.dtype).tiny)
        tangential_electric = (
            electric
            - contract("i,i->", electric, unit_normal, backend="jax") * unit_normal
        )
        slip = (
            -self.permittivity
            * self.zeta_potential
            / self.dynamic_viscosity
            * tangential_electric
        )
        finite = (
            jnp.all(jnp.isfinite(concentration))
            & jnp.all(concentration > 0.0)
            & jnp.isfinite(temperature_)
            & (temperature_ > 0.0)
            & jnp.isfinite(debye)
            & jnp.all(jnp.isfinite(slip))
            & (jnp.abs(normal_norm - 1.0) <= 1.0e-8)
        )
        supported = (
            finite
            & electroneutral
            & (debye_ratio <= self.maximum_debye_ratio)
            & (dukhin <= self.maximum_dukhin_number)
        )
        reasons = jnp.asarray(0, dtype=jnp.uint32)
        reasons = jnp.where(
            finite,
            reasons,
            reasons | jnp.asarray(int(AdmissibilityReason.NONFINITE), jnp.uint32),
        )
        reasons = jnp.where(
            debye_ratio <= self.maximum_debye_ratio,
            reasons,
            reasons | jnp.asarray(int(ThinEDLReason.DEBYE_RATIO_EXCEEDED), jnp.uint32),
        )
        reasons = jnp.where(
            dukhin <= self.maximum_dukhin_number,
            reasons,
            reasons | jnp.asarray(int(ThinEDLReason.DUKHIN_NUMBER_EXCEEDED), jnp.uint32),
        )
        reasons = jnp.where(
            electroneutral,
            reasons,
            reasons | jnp.asarray(int(ThinEDLReason.BULK_NOT_ELECTRONEUTRAL), jnp.uint32),
        )
        margin = jnp.minimum(
            self.maximum_debye_ratio - debye_ratio,
            self.maximum_dukhin_number - dukhin,
        )
        header = AdmissibilityHeader(
            jnp.where(supported, margin, jnp.minimum(margin, -1.0)),
            reasons,
            self.plan_id,
            canonical_fingerprint({"kind": "thin-edl-evidence", "plan": self.plan_id}),
        )
        return ThinEDLSlipEvaluation(
            slip,
            debye,
            debye_ratio,
            jnp.asarray(dukhin, dtype=concentration.dtype),
            charge_defect,
            header,
            _DERIVATIVE_CONTRACT,
            False,
            self.plan_id,
        )


__all__ = [
    "ThinEDLElectroosmoticSlipPlan",
    "ThinEDLReason",
    "ThinEDLSlipEvaluation",
]
