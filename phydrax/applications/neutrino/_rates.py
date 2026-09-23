#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax import ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...measurement import ExposureKind, ExposureRecord
from ._oscillation import OscillationProbabilityResult


class NeutrinoFlux(StrictModule, NonTrainableState):
    energy_edges_gev: Array
    values: Array
    covariance: Array
    exposure: ExposureRecord
    flavor_names: tuple[str, str, str] = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    flux_id: str = eqx.field(static=True)

    def __init__(
        self,
        energy_edges_gev: ArrayLike,
        values: ArrayLike,
        covariance: ArrayLike,
        exposure: ExposureRecord,
        /,
        *,
        source_id: str,
        flavor_names: Sequence[str] = ("electron", "muon", "tau"),
    ):
        edges = np.asarray(energy_edges_gev, dtype=np.float64)
        values_ = np.asarray(values, dtype=np.float64)
        covariance_ = np.asarray(covariance, dtype=np.float64)
        flavors = tuple(str(value).strip() for value in flavor_names)
        if (
            edges.ndim != 1
            or edges.size < 2
            or np.any(~np.isfinite(edges))
            or np.any(np.diff(edges) <= 0.0)
            or values_.shape != (edges.size - 1, 3)
            or covariance_.shape != (values_.size, values_.size)
        ):
            raise ValueError(
                "Neutrino flux edges, values, or covariance shape is invalid."
            )
        if (
            len(flavors) != 3
            or len(set(flavors)) != 3
            or any(not value for value in flavors)
            or np.any(~np.isfinite(values_))
            or np.any(values_ < 0.0)
            or np.any(~np.isfinite(covariance_))
            or not np.allclose(covariance_, covariance_.T)
        ):
            raise ValueError(
                "Neutrino flux content, covariance, or flavor names are invalid."
            )
        if (
            not isinstance(exposure, ExposureRecord)
            or exposure.kind is not ExposureKind.PROTONS_ON_TARGET
        ):
            raise ValueError(
                "Accelerator neutrino flux requires protons-on-target exposure."
            )
        source = str(source_id).strip()
        if not source:
            raise ValueError("source_id is required.")
        self.energy_edges_gev = jnp.asarray(edges)
        self.values = jnp.asarray(values_)
        self.covariance = jnp.asarray(covariance_)
        self.exposure = exposure
        self.flavor_names = flavors
        self.source_id = source
        self.flux_id = canonical_fingerprint(
            {
                "kind": "neutrino-flux",
                "arrays": array_tree_fingerprint((edges, values_, covariance_)),
                "exposure": exposure.exposure_id,
                "flavors": list(flavors),
                "source": source,
            }
        )


class NeutrinoRatePlan(StrictModule, NonTrainableState):
    flux: NeutrinoFlux
    cross_sections: Array
    efficiencies: Array
    migration: Array
    target_count: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        flux: NeutrinoFlux,
        cross_sections: ArrayLike,
        efficiencies: ArrayLike,
        migration: ArrayLike,
        /,
        *,
        target_count: float,
        provider_id: str,
    ):
        if not isinstance(flux, NeutrinoFlux):
            raise TypeError("flux must be NeutrinoFlux.")
        cross = np.asarray(cross_sections, dtype=np.float64)
        efficiency = np.asarray(efficiencies, dtype=np.float64)
        migration_ = np.asarray(migration, dtype=np.float64)
        true_bins = flux.values.shape[0]
        if (
            cross.shape != (true_bins, 3)
            or efficiency.shape != cross.shape
            or migration_.ndim != 2
            or migration_.shape[1] != true_bins
        ):
            raise ValueError(
                "Cross sections, efficiencies, or migration do not align with flux bins."
            )
        if (
            np.any(~np.isfinite(cross))
            or np.any(cross < 0.0)
            or np.any(~np.isfinite(efficiency))
            or np.any((efficiency < 0.0) | (efficiency > 1.0))
            or np.any(~np.isfinite(migration_))
            or np.any(migration_ < 0.0)
            or np.any(np.sum(migration_, axis=0) > 1.0 + 1.0e-12)
        ):
            raise ValueError("Neutrino response fields are outside physical support.")
        target = float(target_count)
        provider = str(provider_id).strip()
        if not np.isfinite(target) or target <= 0.0 or not provider:
            raise ValueError("Neutrino target count and provider are required.")
        self.flux = flux
        self.cross_sections = jnp.asarray(cross)
        self.efficiencies = jnp.asarray(efficiency)
        self.migration = jnp.asarray(migration_)
        self.target_count = target
        self.plan_id = canonical_fingerprint(
            {
                "kind": "neutrino-rate-plan",
                "flux": flux.flux_id,
                "arrays": array_tree_fingerprint((cross, efficiency, migration_)),
                "target_count": target,
                "provider": provider,
            }
        )


class NeutrinoRateResult(StrictModule, NonTrainableState):
    true_flavor_rates: Array
    true_total_rates: Array
    reconstructed_rates: Array
    finite: Array
    valid: Array
    plan_id: str = eqx.field(static=True)


def predict_neutrino_rates(
    plan: NeutrinoRatePlan,
    oscillation: OscillationProbabilityResult,
    /,
) -> NeutrinoRateResult:
    if not isinstance(plan, NeutrinoRatePlan) or not isinstance(
        oscillation, OscillationProbabilityResult
    ):
        raise TypeError("plan and oscillation must use neutrino types.")
    bin_centers = 0.5 * (plan.flux.energy_edges_gev[:-1] + plan.flux.energy_edges_gev[1:])
    if (
        oscillation.probabilities.shape != (plan.flux.values.shape[0], 3, 3)
        or oscillation.energies_gev.shape != bin_centers.shape
    ):
        raise ValueError("Oscillation energy support must align with flux bins.")
    probabilities = eqx.error_if(
        oscillation.probabilities,
        jnp.any(oscillation.energies_gev != bin_centers),
        "Oscillation energies do not match the flux-bin centers.",
    )
    oscillated_flux = ein.contract("es,est->et", plan.flux.values, probabilities)
    rates = (
        oscillated_flux
        * plan.cross_sections
        * plan.efficiencies
        * plan.target_count
        * plan.flux.exposure.value
    )
    total = jnp.sum(rates, axis=1)
    reconstructed = plan.migration @ total
    finite = jnp.all(jnp.isfinite(rates)) & jnp.all(jnp.isfinite(reconstructed))
    valid = (
        finite
        & jnp.all(oscillation.valid)
        & jnp.all(rates >= 0.0)
        & jnp.all(reconstructed >= 0.0)
    )
    return NeutrinoRateResult(rates, total, reconstructed, finite, valid, plan.plan_id)


class NearFarTransferResult(StrictModule, NonTrainableState):
    far_prediction: Array
    finite: Array
    valid: Array
    transfer_id: str = eqx.field(static=True)


def apply_near_far_transfer(
    near_spectrum: ArrayLike, transfer: ArrayLike, /, *, transfer_id: str
) -> NearFarTransferResult:
    near = jnp.asarray(near_spectrum)
    transfer_ = jnp.asarray(transfer, dtype=near.dtype)
    if near.ndim != 1 or transfer_.ndim != 2 or transfer_.shape[1] != near.shape[0]:
        raise ValueError("Near spectrum and transfer operator shapes are incompatible.")
    far = transfer_ @ near
    finite = jnp.all(jnp.isfinite(far))
    valid = (
        finite & jnp.all(near >= 0.0) & jnp.all(transfer_ >= 0.0) & jnp.all(far >= 0.0)
    )
    return NearFarTransferResult(
        jnp.where(valid, far, jnp.nan), finite, valid, str(transfer_id)
    )


__all__ = [
    "NearFarTransferResult",
    "NeutrinoFlux",
    "NeutrinoRatePlan",
    "NeutrinoRateResult",
    "apply_near_far_transfer",
    "predict_neutrino_rates",
]
