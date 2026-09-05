# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ....artifacts import ScientificArtifactEnvelope
from ....ein import contract
from ....qualification import ReferenceArtifactManifest
from ....series import SampledSeries
from ....stochastic.path_sampling import block_mean_uncertainty
from ....units import conversion_factor, UnitDefinition
from .._construct import _identifier
from ..experiments._models import thermal_unfolding_free_energy, ThermodynamicConvention


@dataclass(frozen=True, slots=True)
class ProteinEnsembleComposition:
    """Chemical inventory of the entire box, including solvent/ions/cofactors.

    Component IDs identify chemical species/protonation/isotopes, not merely
    element totals. Equality is required between both states at all temperatures.
    """

    construct_id: str
    chemical_state_id: str
    components: tuple[tuple[str, int], ...]
    parameter_id: str

    def __post_init__(self):
        for value in (self.construct_id, self.chemical_state_id, self.parameter_id):
            _identifier(value, "composition identity")
        if not self.components or len({name for name, _ in self.components}) != len(
            self.components
        ):
            raise ValueError(
                "Composition needs unique chemical species with explicit counts."
            )
        for name, count in self.components:
            _identifier(name, "component")
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ValueError("Composition counts must be positive integers.")
        object.__setattr__(self, "components", tuple(sorted(self.components)))

    def fingerprint(self):
        return canonical_fingerprint(
            {
                "kind": "protein-ensemble-composition",
                "construct": self.construct_id,
                "chemistry": self.chemical_state_id,
                "components": self.components,
                "parameters": self.parameter_id,
            }
        )


@dataclass(frozen=True, slots=True)
class EnthalpyReplica:
    """Equilibrated total H=U+K+pV series with independent-replica evidence.

    Times are physical and their units are explicit. The caller supplies a
    source-supported upper correlation-time bound; every retained nonoverlapping
    block must span at least five such bounds. No automatic inlier deletion or
    diameter filter occurs. This is a conditional sampling-error estimate, not
    force-field/ensemble-representativeness uncertainty.
    """

    series: SampledSeries
    composition: ProteinEnsembleComposition
    basin_id: str
    temperature_kelvin: float
    energy_unit: UnitDefinition
    time_unit: UnitDefinition
    replica_id: str
    independence_id: str
    source: ScientificArtifactEnvelope
    block_size: int
    correlation_time_bound: float
    correlation_evidence_id: str
    equilibration_evidence_id: str
    pressure_condition_id: str

    def __post_init__(self):
        for value in (
            self.basin_id,
            self.replica_id,
            self.independence_id,
            self.correlation_evidence_id,
            self.equilibration_evidence_id,
            self.pressure_condition_id,
        ):
            _identifier(value, "replica evidence")
        if self.source.status != "complete":
            raise ValueError(
                "Only successful physical ensemble artifacts can supply enthalpy."
            )
        values = np.asarray(self.series.values)
        times = np.asarray(self.series.support.broadcast_coordinates())
        if (
            values.ndim != 1
            or times.shape != values.shape
            or not np.all(np.asarray(self.series.sample_valid))
        ):
            raise ValueError(
                "Enthalpy replicas require complete scalar physical-time series; split at missing/reset samples first."
            )
        if (
            not np.all(np.asarray(self.series.support.edge_valid))
            or not np.all(np.isfinite(values))
            or not np.all(np.isfinite(times))
            or not np.all(np.diff(times) > 0)
        ):
            raise ValueError(
                "Replica samples must be finite, uninterrupted and strictly time-ordered."
            )
        from ....units import TIME

        if (
            self.time_unit.dimension != TIME
            or self.series.support.coordinate_kind != "continuous"
        ):
            raise ValueError(
                "Enthalpy correlation evidence requires an explicit physical time unit."
            )
        if self.series.support.coordinate_id != self.time_unit.unit_id:
            raise ValueError(
                "Series time coordinate identity must match its declared time unit."
            )
        if not np.isfinite(self.temperature_kelvin) or self.temperature_kelvin <= 0:
            raise ValueError("Temperature must be positive Kelvin.")
        if (
            isinstance(self.block_size, bool)
            or not isinstance(self.block_size, int)
            or self.block_size < 2
            or values.size % self.block_size
            or values.size // self.block_size < 2
        ):
            raise ValueError(
                "At least two complete nonoverlapping blocks are required; no trailing samples are discarded."
            )
        if (
            not np.isfinite(self.correlation_time_bound)
            or self.correlation_time_bound <= 0
        ):
            raise ValueError(
                "A finite positive independently justified correlation-time bound is required."
            )
        blocks = times.reshape((-1, self.block_size))
        if np.any(blocks[:, -1] - blocks[:, 0] < 5 * self.correlation_time_bound):
            raise ValueError(
                "Blocks must span at least five declared correlation-time bounds."
            )

    def mean_and_variance(self, energy_unit):
        factor = float(conversion_factor(self.energy_unit, energy_unit))
        estimate = block_mean_uncertainty(self.series.values, block_size=self.block_size)
        return estimate.mean * factor, (estimate.standard_error * factor) ** 2


class PairedStateEnthalpyEstimate(StrictModule):
    temperatures: Array
    delta_enthalpy: Array
    standard_errors: Array
    folded_means: Array
    unfolded_means: Array
    replica_counts: Array
    composition_id: str = eqx.field(static=True)
    folded_basin_id: str = eqx.field(static=True)
    unfolded_basin_id: str = eqx.field(static=True)
    energy_unit: UnitDefinition
    source_ids: tuple[str, ...] = eqx.field(static=True)
    estimator_id: str = eqx.field(static=True)


def paired_state_enthalpy(
    folded: tuple[EnthalpyReplica, ...],
    unfolded: tuple[EnthalpyReplica, ...],
    *,
    convention: ThermodynamicConvention,
) -> PairedStateEnthalpyEstimate:
    """PMC-inspired U-minus-F paired-state estimator with matched composition.

    Equal weighting of independent replica means targets the caller's declared
    ensemble. For each state, use the larger of between-replica mean variance
    and propagated block-mean variance, retaining between-conformer variation.
    """
    all_replicas = tuple(folded) + tuple(unfolded)
    if not folded or not unfolded:
        raise ValueError(
            "Both independently defined folded and unfolded ensembles are required."
        )
    if len({value.composition.fingerprint() for value in all_replicas}) != 1:
        raise ValueError(
            "Paired-state composition mismatch: protein, solvent, ions, cofactors, chemistry and parameters must match."
        )
    if len({value.pressure_condition_id for value in all_replicas}) != 1:
        raise ValueError("Enthalpy comparisons require matched pressure conditions.")
    if len({value.independence_id for value in all_replicas}) != len(all_replicas) or len(
        {value.replica_id for value in all_replicas}
    ) != len(all_replicas):
        raise ValueError(
            "Replicas must carry distinct independent realization identities across states and temperatures."
        )
    folded_basins, unfolded_basins = (
        {v.basin_id for v in folded},
        {v.basin_id for v in unfolded},
    )
    if (
        len(folded_basins) != 1
        or len(unfolded_basins) != 1
        or folded_basins == unfolded_basins
    ):
        raise ValueError(
            "Independently declared distinct folded/unfolded basin definitions are required."
        )
    temperatures = sorted({value.temperature_kelvin for value in folded})
    if set(temperatures) != {value.temperature_kelvin for value in unfolded}:
        raise ValueError("Both ensembles must cover identical declared temperatures.")
    means, variances, counts = [], [], []
    for ensemble in (folded, unfolded):
        row_means, row_variances, row_counts = [], [], []
        for temperature in temperatures:
            replicas = tuple(
                value for value in ensemble if value.temperature_kelvin == temperature
            )
            if len(replicas) < 2:
                raise ValueError(
                    "At least two independent replicas per state and temperature are required."
                )
            estimates = tuple(
                value.mean_and_variance(convention.energy_unit) for value in replicas
            )
            replica_means = jnp.stack(tuple(value[0] for value in estimates))
            within = (
                jnp.sum(jnp.stack(tuple(value[1] for value in estimates)))
                / len(replicas) ** 2
            )
            between = jnp.var(replica_means, ddof=1) / len(replicas)
            row_means.append(jnp.mean(replica_means))
            row_variances.append(jnp.maximum(within, between))
            row_counts.append(len(replicas))
        means.append(jnp.stack(row_means))
        variances.append(jnp.stack(row_variances))
        counts.append(row_counts)
    identity = canonical_fingerprint(
        {
            "kind": "pmc-paired-state-enthalpy",
            "replicas": [
                (
                    value.replica_id,
                    value.source.artifact_id,
                    value.series.series_id,
                    array_tree_fingerprint(
                        (value.series.values, value.series.support.coordinates)
                    ),
                    value.temperature_kelvin,
                    value.block_size,
                    value.correlation_time_bound,
                    value.correlation_evidence_id,
                    value.energy_unit.unit_id,
                    value.time_unit.unit_id,
                    value.pressure_condition_id,
                )
                for value in all_replicas
            ],
            "composition": all_replicas[0].composition.fingerprint(),
            "unit": convention.energy_unit.unit_id,
            "sign": "unfolded-minus-folded",
        }
    )
    return PairedStateEnthalpyEstimate(
        jnp.asarray(temperatures),
        means[1] - means[0],
        jnp.sqrt(variances[0] + variances[1]),
        means[0],
        means[1],
        jnp.asarray(counts),
        all_replicas[0].composition.fingerprint(),
        next(iter(folded_basins)),
        next(iter(unfolded_basins)),
        convention.energy_unit,
        tuple(value.source.artifact_id for value in all_replicas),
        identity,
    )


class HeatCapacitySlopeEstimate(StrictModule):
    reference_temperature: Array
    reference_enthalpy: Array
    delta_heat_capacity: Array
    covariance: Array
    residuals: Array
    temperature_interval: Array
    source: PairedStateEnthalpyEstimate


def fit_heat_capacity_slope(
    estimate: PairedStateEnthalpyEstimate,
) -> HeatCapacitySlopeEstimate:
    """Finite-interval linear ΔH(T) fit; residuals expose constant-ΔCp inadequacy.

    Covariance propagates independent per-temperature sampling uncertainties; it
    does not include model-form error and is not a posterior interval.
    """
    t, h, se = estimate.temperatures, estimate.delta_enthalpy, estimate.standard_errors
    if t.size < 3:
        raise ValueError(
            "At least three temperatures are required to assess a linear heat-capacity slope."
        )
    center = jnp.mean(t)
    x = t - center
    slope_weights = x / jnp.sum(x * x)
    mean_weights = jnp.ones_like(t) / t.size
    weights = jnp.stack((mean_weights, slope_weights))
    coefficients = contract("ij,j->i", weights, h)
    covariance = contract("in,jn,n->ij", weights, weights, se**2)
    residuals = h - coefficients[0] - coefficients[1] * x
    return HeatCapacitySlopeEstimate(
        center,
        coefficients[0],
        coefficients[1],
        covariance,
        residuals,
        jnp.asarray([t[0], t[-1]]),
        estimate,
    )


class ExperimentallyClosedFreeEnergy(StrictModule):
    temperatures: Array
    delta_free_energy: Array
    standard_errors: Array
    valid: Array
    experimental_dependencies: tuple[str, ...] = eqx.field(static=True)
    closure_kind: str = eqx.field(static=True)
    source_estimator_id: str = eqx.field(static=True)


def close_free_energy_at_reference(
    fit: HeatCapacitySlopeEstimate,
    temperatures,
    *,
    reference_temperature: float,
    reference_delta_g: float,
    experimental_covariance,
    reference: ReferenceArtifactManifest,
    closure_kind: str,
    commercial_use=False,
) -> ExperimentallyClosedFreeEnergy:
    """Close ΔG using measured (T_ref, ΔG_ref), with their 2×2 covariance.

    ``closure_kind`` names either measured melting-temperature (ΔG=0) or an
    experimentally closed thermodynamic cycle (e.g. ligand binding). A ligand
    Kd alone is NOT a folding free energy: the caller must supply the closed
    cycle's ΔG and uncertainty. Experimental/MD sources are assumed independent.
    Constant ΔCp is admitted only inside the sampled temperature interval.
    """
    reference.require_rights(commercial_use=commercial_use)
    reference.require_uncertainty()
    if closure_kind not in (
        "measured-melting-temperature",
        "experimental-thermodynamic-cycle",
    ):
        raise ValueError("Declare an experimental melting or closed-cycle dependency.")
    if closure_kind == "measured-melting-temperature" and reference_delta_g != 0:
        raise ValueError("The thermodynamic melting closure requires ΔG(Tm)=0.")
    lower, upper = np.asarray(fit.temperature_interval)
    if (
        not np.isfinite(reference_temperature)
        or not lower <= reference_temperature <= upper
        or not np.isfinite(reference_delta_g)
    ):
        raise ValueError(
            "Experimental closure must lie inside the enthalpy model's sampled interval."
        )
    covariance = np.asarray(experimental_covariance, dtype=float)
    if (
        covariance.shape != (2, 2)
        or not np.all(np.isfinite(covariance))
        or not np.allclose(covariance, covariance.T)
        or np.any(np.diag(covariance) < 0)
        or covariance[0, 0] * covariance[1, 1] < covariance[0, 1] ** 2
    ):
        raise ValueError(
            "Experimental (temperature, ΔG) covariance must be symmetric positive semidefinite."
        )
    t = jnp.asarray(temperatures)
    parameters = jnp.asarray(
        [
            fit.reference_enthalpy,
            fit.delta_heat_capacity,
            reference_temperature,
            reference_delta_g,
        ]
    )
    full_covariance = (
        jnp.zeros((4, 4), dtype=parameters.dtype)
        .at[:2, :2]
        .set(fit.covariance)
        .at[2:, 2:]
        .set(covariance)
    )

    def evaluate(p):
        h_ref, cp, t_ref, g_ref = p
        h_at_closure = h_ref + cp * (t_ref - fit.reference_temperature)
        return thermal_unfolding_free_energy(
            jnp.asarray([g_ref, h_at_closure, cp, 0.0, 0.0]), t, 0.0, t_ref
        )

    result = evaluate(parameters)
    derivative = jax.jacfwd(evaluate)(parameters)
    variance = contract("...i,ij,...j->...", derivative, full_covariance, derivative)
    valid = jnp.isfinite(t) & (t >= lower) & (t <= upper)
    return ExperimentallyClosedFreeEnergy(
        t,
        jnp.where(valid, result, jnp.nan),
        jnp.where(valid, jnp.sqrt(jnp.maximum(variance, 0)), jnp.nan),
        valid,
        (reference.manifest_id,),
        closure_kind,
        fit.source.estimator_id,
    )


__all__ = [
    "ProteinEnsembleComposition",
    "EnthalpyReplica",
    "PairedStateEnthalpyEstimate",
    "HeatCapacitySlopeEstimate",
    "ExperimentallyClosedFreeEnergy",
    "paired_state_enthalpy",
    "fit_heat_capacity_slope",
    "close_free_energy_at_reference",
]
