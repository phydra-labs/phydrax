#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from .._core import SensitiveHitBank
from ._geometry import CalorimeterGeometry


class CalorimeterTruth(StrictModule, NonTrainableState):
    incident_energy: Array
    source_deposited_energy: Array
    cell_energies: Array
    outside_energy: Array
    unmapped_energy: Array
    dead_energy: Array
    rejected_energy: Array
    leakage_energy: Array
    leakage_known: Array
    deposited_residual: Array
    incident_residual: Array
    valid: Array
    event_ids: Array
    geometry_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return jnp.all(self.valid)


def route_calorimeter_hits(
    geometry: CalorimeterGeometry,
    hits: SensitiveHitBank,
    incident_energy: ArrayLike,
    source_deposited_energy: ArrayLike,
    /,
    *,
    outside_energy: ArrayLike | None = None,
    rejected_energy: ArrayLike | None = None,
    leakage_energy: ArrayLike | None = None,
    leakage_known: ArrayLike | None = None,
    tolerance: float = 1.0e-10,
    source_id: str,
) -> CalorimeterTruth:
    """Route hit energy into cells and retain every known ledger remainder."""
    if not isinstance(geometry, CalorimeterGeometry) or not isinstance(
        hits, SensitiveHitBank
    ):
        raise TypeError("geometry and hits must use calorimeter/detector types.")
    if geometry.conditions_id != hits.conditions_id:
        raise ValueError("Calorimeter geometry and hits use different conditions.")
    incident = jnp.asarray(incident_energy, dtype=hits.energies.dtype)
    deposited_source = jnp.asarray(source_deposited_energy, dtype=hits.energies.dtype)
    event_count = hits.event_ids.shape[0]
    expected = (event_count,)
    if incident.shape != expected or deposited_source.shape != expected:
        raise ValueError("Incident and deposited source energies must align with events.")
    outside = (
        jnp.zeros(expected, dtype=incident.dtype)
        if outside_energy is None
        else jnp.asarray(outside_energy, dtype=incident.dtype)
    )
    rejected = (
        jnp.zeros(expected, dtype=incident.dtype)
        if rejected_energy is None
        else jnp.asarray(rejected_energy, dtype=incident.dtype)
    )
    leakage = (
        jnp.full(expected, jnp.nan, dtype=incident.dtype)
        if leakage_energy is None
        else jnp.asarray(leakage_energy, dtype=incident.dtype)
    )
    leakage_known_ = (
        jnp.zeros(expected, dtype=jnp.bool_)
        if leakage_known is None
        else jnp.asarray(leakage_known, dtype=jnp.bool_)
    )
    if any(
        value.shape != expected for value in (outside, rejected, leakage, leakage_known_)
    ):
        raise ValueError("Calorimeter event ledgers must align with events.")
    hit_valid = hits.active & hits.valid
    matches = (hits.channel_ids[..., None] == geometry.channel_ids) & geometry.active
    matched = jnp.any(matches, axis=-1)
    live_matches = matches & ~geometry.dead
    dead_matches = matches & geometry.dead
    energy = jnp.where(hit_valid, hits.energies, 0.0)
    cell_energies = jnp.sum(energy[..., None] * live_matches, axis=1)
    dead = jnp.sum(energy[..., None] * dead_matches, axis=(1, 2))
    unmapped = jnp.sum(jnp.where(hit_valid & ~matched, hits.energies, 0.0), axis=1)
    routed = jnp.sum(cell_energies, axis=1)
    deposited_residual = deposited_source - (
        routed + outside + unmapped + dead + rejected
    )
    incident_residual = jnp.where(
        leakage_known_,
        incident - (deposited_source + leakage),
        jnp.nan,
    )
    scale = jnp.maximum(jnp.abs(deposited_source), 1.0)
    deposited_valid = jnp.abs(deposited_residual) <= float(tolerance) * scale
    incident_valid = ~leakage_known_ | (
        jnp.isfinite(leakage)
        & (leakage >= 0.0)
        & (
            jnp.abs(incident_residual)
            <= float(tolerance) * jnp.maximum(jnp.abs(incident), 1.0)
        )
    )
    nonnegative = (
        (incident >= 0.0)
        & (deposited_source >= 0.0)
        & (outside >= 0.0)
        & (rejected >= 0.0)
    )
    finite = (
        jnp.isfinite(incident)
        & jnp.isfinite(deposited_source)
        & jnp.isfinite(outside)
        & jnp.isfinite(rejected)
        & jnp.all(jnp.isfinite(cell_energies), axis=1)
    )
    source = str(source_id).strip()
    if not source:
        raise ValueError("source_id must be non-empty.")
    return CalorimeterTruth(
        incident,
        deposited_source,
        cell_energies,
        outside,
        unmapped,
        dead,
        rejected,
        leakage,
        leakage_known_,
        deposited_residual,
        incident_residual,
        deposited_valid & incident_valid & nonnegative & finite,
        hits.event_ids,
        geometry.geometry_id,
        source,
    )


__all__ = ["CalorimeterTruth", "route_calorimeter_hits"]
