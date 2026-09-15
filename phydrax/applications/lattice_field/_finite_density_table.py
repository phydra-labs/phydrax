#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._finite_density import (
    ChemicalChargeConvention,
    FiniteDensityDomain,
    FiniteDensitySourceKind,
    FiniteDensityStatus,
)
from ._finite_density_taylor import evaluate_taylor_eos, PreparedTaylorEOS


class EOSGridPlan(StrictModule, NonTrainableState):
    temperatures: Array
    baryon_chemical_potentials: Array
    convention: ChemicalChargeConvention
    domain: FiniteDensityDomain
    constraint_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        temperatures: ArrayLike,
        baryon_chemical_potentials: ArrayLike,
        /,
        *,
        convention: ChemicalChargeConvention,
        domain: FiniteDensityDomain,
        constraint_id: str = "muQ=muS=0",
        maximum_cells: int = 1_000_000,
    ):
        temperatures_ = np.asarray(temperatures, dtype=float)
        baryon = np.asarray(baryon_chemical_potentials, dtype=float)
        if (
            temperatures_.ndim != 1
            or baryon.ndim != 1
            or temperatures_.size < 2
            or baryon.size < 2
            or np.any(~np.isfinite(temperatures_))
            or np.any(~np.isfinite(baryon))
            or np.any(np.diff(temperatures_) <= 0.0)
            or np.any(np.diff(baryon) <= 0.0)
        ):
            raise ValueError("EoS grid axes must be finite increasing vectors.")
        if temperatures_.size * baryon.size > int(maximum_cells):
            raise ValueError("EoS grid exceeds maximum_cells.")
        if not isinstance(convention, ChemicalChargeConvention) or not isinstance(
            domain, FiniteDensityDomain
        ):
            raise TypeError("EoS grid convention and domain must be explicit.")
        constraint = str(constraint_id).strip()
        if constraint != "muQ=muS=0":
            raise ValueError(
                "This table builder currently supports the explicit muQ=muS=0 path."
            )
        self.temperatures = jnp.asarray(temperatures_)
        self.baryon_chemical_potentials = jnp.asarray(baryon)
        self.convention = convention
        self.domain = domain
        self.constraint_id = constraint
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-density-eos-grid-plan",
                "axes": array_tree_fingerprint((temperatures_, baryon)),
                "convention": convention.convention_id,
                "domain": domain.domain_id,
                "constraint": constraint,
            }
        )


class FiniteDensityEOSTable(StrictModule, NonTrainableState):
    plan: EOSGridPlan
    pressure_over_temperature4: Array
    densities_over_temperature3: Array
    entropy_over_temperature3: Array
    energy_over_temperature4: Array
    susceptibility_matrix: Array
    thermodynamic_identity_residual: Array
    derivative_valid: Array
    valid: Array
    source_kind: FiniteDensitySourceKind = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    table_id: str = eqx.field(static=True)


class EOSQualification(StrictModule, NonTrainableState):
    finite: Array
    stable: Array
    causal: Array
    speed_of_sound_squared: Array
    maximum_identity_residual: Array
    qualified_cells: Array
    passed: Array
    table_id: str = eqx.field(static=True)


def build_taylor_eos_table(
    prepared: PreparedTaylorEOS,
    plan: EOSGridPlan,
    /,
) -> FiniteDensityEOSTable:
    """Materialize a masked static μQ=μS=0 table from one Taylor potential."""
    if not isinstance(prepared, PreparedTaylorEOS) or not isinstance(plan, EOSGridPlan):
        raise TypeError("prepared and plan must use finite-density table types.")
    if (
        prepared.estimate.convention.convention_id != plan.convention.convention_id
        or prepared.estimate.domain.domain_id != plan.domain.domain_id
    ):
        raise ValueError("Taylor source and EoS grid convention/domain differ.")

    def one_temperature(temperature):
        return jax.vmap(
            lambda baryon: evaluate_taylor_eos(
                prepared,
                temperature,
                jnp.asarray([baryon, 0.0, 0.0]),
            )
        )(plan.baryon_chemical_potentials)

    results = jax.vmap(one_temperature)(plan.temperatures)
    valid = results.successful & results.in_domain
    table_id = canonical_fingerprint(
        {
            "kind": "finite-density-taylor-eos-table",
            "plan": plan.plan_id,
            "source": prepared.prepared_id,
        }
    )
    return FiniteDensityEOSTable(
        plan,
        results.pressure_over_temperature4,
        results.densities_over_temperature3,
        results.entropy_over_temperature3,
        results.energy_over_temperature4,
        results.susceptibility_matrix,
        results.thermodynamic_identity_residual,
        results.derivative_valid,
        valid,
        prepared.estimate.source_kind,
        prepared.prepared_id,
        table_id,
    )


def qualify_eos_table(
    table: FiniteDensityEOSTable,
    /,
    *,
    identity_tolerance: float = 1.0e-9,
    stability_tolerance: float = 1.0e-10,
) -> EOSQualification:
    """Check finite thermodynamics, susceptibility stability, and grid-line causality."""
    if not isinstance(table, FiniteDensityEOSTable):
        raise TypeError("table must be FiniteDensityEOSTable.")
    finite = (
        jnp.isfinite(table.pressure_over_temperature4)
        & jnp.isfinite(table.entropy_over_temperature3)
        & jnp.isfinite(table.energy_over_temperature4)
        & jnp.all(jnp.isfinite(table.densities_over_temperature3), axis=-1)
        & jnp.all(jnp.isfinite(table.susceptibility_matrix), axis=(-1, -2))
    )
    symmetric = 0.5 * (
        table.susceptibility_matrix + jnp.swapaxes(table.susceptibility_matrix, -1, -2)
    )
    eigenvalues = jnp.linalg.eigvalsh(symmetric)
    stable = (jnp.min(eigenvalues, axis=-1) >= -float(stability_tolerance)) & (
        table.entropy_over_temperature3 >= -float(stability_tolerance)
    )
    temperature = table.plan.temperatures[:, None]
    pressure = temperature**4 * table.pressure_over_temperature4
    energy = temperature**4 * table.energy_over_temperature4
    dp = pressure[1:] - pressure[:-1]
    de = energy[1:] - energy[:-1]
    interval_speed = dp / jnp.where(
        jnp.abs(de) > jnp.finfo(de.dtype).tiny,
        de,
        jnp.nan,
    )
    speed = jnp.concatenate((interval_speed[:1], interval_speed), axis=0)
    causal = (
        jnp.isfinite(speed)
        & (speed >= -float(stability_tolerance))
        & (speed <= 1.0 + float(stability_tolerance))
    )
    identity = jnp.abs(table.thermodynamic_identity_residual)
    identity_valid = identity <= float(identity_tolerance)
    qualified = (
        table.valid & table.derivative_valid & finite & stable & causal & identity_valid
    )
    return EOSQualification(
        finite,
        stable,
        causal,
        speed,
        jnp.max(jnp.where(table.valid, identity, 0.0)),
        qualified,
        jnp.all(qualified | ~table.valid) & jnp.any(table.valid),
        table.table_id,
    )


class EOSTableEvaluation(StrictModule, NonTrainableState):
    pressure_over_temperature4: Array
    densities_over_temperature3: Array
    entropy_over_temperature3: Array
    energy_over_temperature4: Array
    derivative_valid: Array
    valid: Array
    status: Array
    table_id: str = eqx.field(static=True)


def evaluate_eos_table(
    table: FiniteDensityEOSTable,
    temperature: ArrayLike,
    baryon_chemical_potential: ArrayLike,
    /,
) -> EOSTableEvaluation:
    """Bilinearly evaluate qualified source cells without extrapolation or hole filling."""
    if not isinstance(table, FiniteDensityEOSTable):
        raise TypeError("table must be FiniteDensityEOSTable.")
    temperature_ = jnp.asarray(temperature, dtype=table.plan.temperatures.dtype).reshape(
        ()
    )
    baryon = jnp.asarray(baryon_chemical_potential, dtype=temperature_.dtype).reshape(())
    temperatures = table.plan.temperatures
    chemical = table.plan.baryon_chemical_potentials
    ti = jnp.clip(
        jnp.searchsorted(temperatures, temperature_, side="right") - 1,
        0,
        temperatures.size - 2,
    )
    mi = jnp.clip(
        jnp.searchsorted(chemical, baryon, side="right") - 1, 0, chemical.size - 2
    )
    ft = (temperature_ - temperatures[ti]) / (temperatures[ti + 1] - temperatures[ti])
    fm = (baryon - chemical[mi]) / (chemical[mi + 1] - chemical[mi])

    def interpolate(values):
        return (
            (1.0 - ft) * (1.0 - fm) * values[ti, mi]
            + ft * (1.0 - fm) * values[ti + 1, mi]
            + (1.0 - ft) * fm * values[ti, mi + 1]
            + ft * fm * values[ti + 1, mi + 1]
        )

    corners_valid = (
        table.valid[ti, mi]
        & table.valid[ti + 1, mi]
        & table.valid[ti, mi + 1]
        & table.valid[ti + 1, mi + 1]
    )
    corners_derivative = (
        table.derivative_valid[ti, mi]
        & table.derivative_valid[ti + 1, mi]
        & table.derivative_valid[ti, mi + 1]
        & table.derivative_valid[ti + 1, mi + 1]
    )
    inside = (
        (temperature_ >= temperatures[0])
        & (temperature_ <= temperatures[-1])
        & (baryon >= chemical[0])
        & (baryon <= chemical[-1])
    )
    valid = inside & corners_valid
    status = jnp.where(
        valid, int(FiniteDensityStatus.SUCCESS), int(FiniteDensityStatus.OUTSIDE_DOMAIN)
    )
    pressure = interpolate(table.pressure_over_temperature4)
    densities = interpolate(table.densities_over_temperature3)
    entropy = interpolate(table.entropy_over_temperature3)
    energy = interpolate(table.energy_over_temperature4)
    return EOSTableEvaluation(
        jnp.where(valid, pressure, jnp.nan),
        jnp.where(valid, densities, jnp.nan),
        jnp.where(valid, entropy, jnp.nan),
        jnp.where(valid, energy, jnp.nan),
        valid & corners_derivative,
        valid,
        status.astype(jnp.int32),
        table.table_id,
    )


def eos_table_metadata(
    table: FiniteDensityEOSTable, qualification: EOSQualification, /
) -> dict[str, object]:
    """Return canonical internal metadata; array persistence remains lifecycle-owned."""
    if table.table_id != qualification.table_id:
        raise ValueError("EoS table and qualification identities differ.")
    if not bool(qualification.passed):
        raise ValueError("Unqualified EoS tables cannot be exported.")
    return {
        "kind": "finite-density-eos-table",
        "table_id": table.table_id,
        "plan_id": table.plan.plan_id,
        "source_id": table.source_id,
        "source_kind": table.source_kind.value,
        "convention_id": table.plan.convention.convention_id,
        "domain_id": table.plan.domain.domain_id,
        "constraint_id": table.plan.constraint_id,
        "array_fingerprint": array_tree_fingerprint(
            {
                "temperatures": table.plan.temperatures,
                "baryon_chemical_potentials": table.plan.baryon_chemical_potentials,
                "pressure_over_temperature4": table.pressure_over_temperature4,
                "densities_over_temperature3": table.densities_over_temperature3,
                "entropy_over_temperature3": table.entropy_over_temperature3,
                "energy_over_temperature4": table.energy_over_temperature4,
                "valid": table.valid,
            }
        ),
    }


__all__ = [
    "EOSGridPlan",
    "EOSQualification",
    "EOSTableEvaluation",
    "FiniteDensityEOSTable",
    "build_taylor_eos_table",
    "eos_table_metadata",
    "evaluate_eos_table",
    "qualify_eos_table",
]
