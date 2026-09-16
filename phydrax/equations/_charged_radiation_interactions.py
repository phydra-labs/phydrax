#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..qualification import ReferenceArtifactManifest


class ChargedRadiationParticleKind(IntEnum):
    ELECTRON = 0
    POSITRON = 1


class ChargedRadiationMaterialEvaluation(StrictModule):
    stopping_power: Array
    scattering_power: Array
    bremsstrahlung_rate: Array
    supported: Array
    finite: Array
    successful: Array
    library_id: str = eqx.field(static=True)


class ChargedRadiationMaterialLibrary(StrictModule, NonTrainableState):
    """Governed condensed-history coefficients in eV/m, rad²/m, and 1/m."""

    energy_ev: Array
    stopping_power_ev_m: Array
    scattering_power_rad2_m: Array
    bremsstrahlung_rate_m: Array
    material_ids: tuple[str, ...] = eqx.field(static=True)
    manifest: ReferenceArtifactManifest = eqx.field(static=True)
    library_id: str = eqx.field(static=True)

    def __init__(
        self,
        energy_ev: ArrayLike,
        stopping_power_ev_m: ArrayLike,
        scattering_power_rad2_m: ArrayLike,
        bremsstrahlung_rate_m: ArrayLike,
        material_ids: tuple[str, ...],
        manifest: ReferenceArtifactManifest,
        /,
        *,
        commercial_use: bool = False,
        export: bool = False,
    ):
        energy = np.asarray(energy_ev, dtype=float)
        stopping = np.asarray(stopping_power_ev_m, dtype=float)
        scattering = np.asarray(scattering_power_rad2_m, dtype=float)
        brems = np.asarray(bremsstrahlung_rate_m, dtype=float)
        materials = tuple(str(value).strip() for value in material_ids)
        if not isinstance(manifest, ReferenceArtifactManifest):
            raise TypeError("Charged radiation material data require a manifest.")
        manifest.require_rights(commercial_use=commercial_use, export=export)
        shape = (len(materials), energy.size)
        if (
            energy.ndim != 1
            or energy.size < 2
            or np.any(~np.isfinite(energy))
            or np.any(energy <= 0.0)
            or np.any(np.diff(energy) <= 0.0)
            or not materials
            or len(set(materials)) != len(materials)
            or any(not value for value in materials)
            or stopping.shape != shape
            or scattering.shape != shape
            or brems.shape != shape
            or np.any(~np.isfinite(stopping))
            or np.any(stopping <= 0.0)
            or np.any(~np.isfinite(scattering))
            or np.any(scattering < 0.0)
            or np.any(~np.isfinite(brems))
            or np.any(brems < 0.0)
        ):
            raise ValueError("Charged radiation material tables are invalid.")
        self.energy_ev = jnp.asarray(energy)
        self.stopping_power_ev_m = jnp.asarray(stopping)
        self.scattering_power_rad2_m = jnp.asarray(scattering)
        self.bremsstrahlung_rate_m = jnp.asarray(brems)
        self.material_ids = materials
        self.manifest = manifest
        self.library_id = canonical_fingerprint(
            {
                "kind": "charged-radiation-condensed-history-library",
                "energy_ev": array_tree_fingerprint(energy),
                "stopping_power_ev_m": array_tree_fingerprint(stopping),
                "scattering_power_rad2_m": array_tree_fingerprint(scattering),
                "bremsstrahlung_rate_m": array_tree_fingerprint(brems),
                "materials": materials,
                "manifest": manifest.manifest_id,
            }
        )

    @property
    def material_count(self) -> int:
        return len(self.material_ids)

    def evaluate(
        self, material_index: ArrayLike, kinetic_energy_ev: ArrayLike, /
    ) -> ChargedRadiationMaterialEvaluation:
        material = jnp.asarray(material_index, dtype=jnp.int32)
        energy = jnp.asarray(kinetic_energy_ev, dtype=self.energy_ev.dtype)
        shape = jnp.broadcast_shapes(material.shape, energy.shape)
        material = jnp.broadcast_to(material, shape)
        energy = jnp.broadcast_to(energy, shape)
        supported = (
            (material >= 0)
            & (material < self.material_count)
            & jnp.isfinite(energy)
            & (energy >= self.energy_ev[0])
            & (energy <= self.energy_ev[-1])
        )
        safe_material = jnp.clip(material, 0, self.material_count - 1)
        safe_energy = jnp.clip(energy, self.energy_ev[0], self.energy_ev[-1])
        upper = jnp.clip(
            jnp.searchsorted(self.energy_ev, safe_energy, side="right"),
            1,
            self.energy_ev.size - 1,
        )
        lower = upper - 1
        fraction = (safe_energy - self.energy_ev[lower]) / (
            self.energy_ev[upper] - self.energy_ev[lower]
        )

        def interpolate(table):
            left = table[safe_material, lower]
            right = table[safe_material, upper]
            return left + fraction * (right - left)

        stopping = interpolate(self.stopping_power_ev_m)
        scattering = interpolate(self.scattering_power_rad2_m)
        brems = interpolate(self.bremsstrahlung_rate_m)
        finite = jnp.isfinite(stopping) & jnp.isfinite(scattering) & jnp.isfinite(brems)
        successful = (
            supported & finite & (stopping > 0.0) & (scattering >= 0.0) & (brems >= 0.0)
        )
        return ChargedRadiationMaterialEvaluation(
            stopping, scattering, brems, supported, finite, successful, self.library_id
        )


__all__ = [
    "ChargedRadiationMaterialEvaluation",
    "ChargedRadiationMaterialLibrary",
    "ChargedRadiationParticleKind",
]
