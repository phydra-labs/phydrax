#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...qualification import ReferenceArtifactManifest
from ...units import conversion_factor, ELECTRONVOLT, UnitDefinition


class PhotonSourceBatch(StrictModule, NonTrainableState):
    history_ids: Array
    origins: Array
    directions: Array
    energies: Array
    weights: Array
    finite: Array
    successful: Array
    source_id: str = eqx.field(static=True)


class AliasSpectrumPlan(StrictModule, NonTrainableState):
    energies: Array
    probability: Array
    alias: Array
    energy_unit: UnitDefinition = eqx.field(static=True)
    manifest: ReferenceArtifactManifest = eqx.field(static=True)
    spectrum_id: str = eqx.field(static=True)

    def __init__(
        self,
        energies: ArrayLike,
        probabilities: ArrayLike,
        energy_unit: UnitDefinition,
        manifest: ReferenceArtifactManifest,
        /,
        *,
        commercial_use: bool = False,
        export: bool = False,
    ):
        energy = np.asarray(energies, dtype=np.float64)
        probability = np.asarray(probabilities, dtype=np.float64)
        if not isinstance(energy_unit, UnitDefinition):
            raise TypeError("energy_unit must be UnitDefinition.")
        conversion_factor(energy_unit, ELECTRONVOLT)
        if not isinstance(manifest, ReferenceArtifactManifest):
            raise TypeError("Photon spectra require a reference manifest.")
        manifest.require_rights(commercial_use=commercial_use, export=export)
        if (
            energy.ndim != 1
            or energy.size < 1
            or probability.shape != energy.shape
            or np.any(~np.isfinite(energy))
            or np.any(energy <= 0.0)
            or np.any(~np.isfinite(probability))
            or np.any(probability < 0.0)
            or np.sum(probability) <= 0.0
        ):
            raise ValueError("Photon spectrum energies or probabilities are invalid.")
        normalized = probability / np.sum(probability)
        count = energy.size
        scaled = normalized * count
        small = [index for index, value in enumerate(scaled) if value < 1.0]
        large = [index for index, value in enumerate(scaled) if value >= 1.0]
        acceptance = np.ones((count,), dtype=np.float64)
        alias = np.arange(count, dtype=np.int32)
        while small and large:
            low = small.pop()
            high = large.pop()
            acceptance[low] = scaled[low]
            alias[low] = high
            scaled[high] = scaled[high] - (1.0 - scaled[low])
            (small if scaled[high] < 1.0 else large).append(high)
        self.energies = jnp.asarray(energy)
        self.probability = jnp.asarray(acceptance)
        self.alias = jnp.asarray(alias)
        self.energy_unit = energy_unit
        self.manifest = manifest
        self.spectrum_id = canonical_fingerprint(
            {
                "kind": "walker-alias-photon-spectrum",
                "energies": array_tree_fingerprint(energy),
                "probabilities": array_tree_fingerprint(normalized),
                "energy_unit": energy_unit.unit_id,
                "manifest": manifest.manifest_id,
            }
        )

    def sample(self, key: Array, history_ids: ArrayLike, /) -> Array:
        history = jnp.asarray(history_ids, dtype=jnp.uint32)

        def one(identifier):
            first = jr.uniform(jr.fold_in(key, 2 * identifier))
            second = jr.uniform(jr.fold_in(key, 2 * identifier + 1))
            column = jnp.minimum(
                jnp.floor(first * self.energies.size).astype(jnp.int32),
                self.energies.size - 1,
            )
            selected = jnp.where(
                second < self.probability[column], column, self.alias[column]
            )
            return self.energies[selected]

        return jax.vmap(one)(history.reshape((-1,))).reshape(history.shape)


class DiagnosticXRaySourcePlan(StrictModule, NonTrainableState):
    position: Array
    axis: Array
    cone_half_angle: float = eqx.field(static=True)
    spectrum: AliasSpectrumPlan
    source_id: str = eqx.field(static=True)

    def __init__(
        self,
        position: ArrayLike,
        axis: ArrayLike,
        spectrum: AliasSpectrumPlan,
        /,
        *,
        cone_half_angle: float,
        source_id: str,
    ):
        position_ = np.asarray(position, dtype=np.float64)
        axis_ = np.asarray(axis, dtype=np.float64)
        angle = float(cone_half_angle)
        identifier = str(source_id).strip()
        norm = np.linalg.norm(axis_)
        if (
            position_.shape != (3,)
            or axis_.shape != (3,)
            or np.any(~np.isfinite(position_))
            or np.any(~np.isfinite(axis_))
            or norm <= 0.0
            or not isfinite(angle)
            or not 0.0 <= angle < 0.5 * np.pi
            or not isinstance(spectrum, AliasSpectrumPlan)
            or not identifier
        ):
            raise ValueError("Diagnostic X-ray source geometry or identity is invalid.")
        self.position = jnp.asarray(position_)
        self.axis = jnp.asarray(axis_ / norm)
        self.cone_half_angle = angle
        self.spectrum = spectrum
        self.source_id = canonical_fingerprint(
            {
                "kind": "diagnostic-xray-cone-source",
                "declared_id": identifier,
                "position": position_.tolist(),
                "axis": (axis_ / norm).tolist(),
                "cone_half_angle": angle,
                "spectrum": spectrum.spectrum_id,
            }
        )

    def sample(
        self, key: Array, history_count: int, /, *, first_history_id: int = 0
    ) -> PhotonSourceBatch:
        count = int(history_count)
        first = int(first_history_id)
        if count < 1 or first < 0:
            raise ValueError("Photon history count and first ID are invalid.")
        history = jnp.arange(first, first + count, dtype=jnp.uint32)
        energies = self.spectrum.sample(jr.fold_in(key, 0), history)
        cosine_minimum = jnp.cos(self.cone_half_angle)

        def direction(identifier):
            cosine = cosine_minimum + (1.0 - cosine_minimum) * jr.uniform(
                jr.fold_in(key, 3 * identifier + 1)
            )
            azimuth = 2.0 * jnp.pi * jr.uniform(jr.fold_in(key, 3 * identifier + 2))
            z = self.axis
            reference = jnp.where(
                jnp.abs(z[2]) < 0.9,
                jnp.asarray((0.0, 0.0, 1.0), dtype=z.dtype),
                jnp.asarray((1.0, 0.0, 0.0), dtype=z.dtype),
            )
            x = jnp.cross(reference, z)
            x = x / jnp.linalg.norm(x)
            y = jnp.cross(z, x)
            sine = jnp.sqrt(jnp.maximum(1.0 - cosine**2, 0.0))
            return cosine * z + sine * (jnp.cos(azimuth) * x + jnp.sin(azimuth) * y)

        directions = jax.vmap(direction)(history)
        origins = jnp.broadcast_to(self.position, (count, 3))
        weights = jnp.ones((count,), dtype=energies.dtype)
        finite = (
            jnp.all(jnp.isfinite(origins))
            & jnp.all(jnp.isfinite(directions))
            & jnp.all(jnp.isfinite(energies))
        )
        return PhotonSourceBatch(
            history,
            origins,
            directions,
            energies,
            weights,
            finite,
            finite,
            self.source_id,
        )


__all__ = ["AliasSpectrumPlan", "DiagnosticXRaySourcePlan", "PhotonSourceBatch"]
