#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._aerothermal_material import ConservativeRecessionRemapPlan


class HighEnthalpyAMREvidence(StrictModule):
    indicators: Array
    refine_mask: Array
    coarsen_mask: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class HighEnthalpyAMRIndicatorPlan(StrictModule, NonTrainableState):
    """Named vector indicators with per-channel refine/coarsen thresholds."""

    indicator_names: tuple[str, ...] = eqx.field(static=True)
    refine_thresholds: Array
    coarsen_thresholds: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        indicator_names: Sequence[str],
        refine_thresholds: ArrayLike,
        coarsen_thresholds: ArrayLike,
        /,
    ):
        names = tuple(str(value) for value in indicator_names)
        refine = np.asarray(refine_thresholds, dtype=float)
        coarsen = np.asarray(coarsen_thresholds, dtype=float)
        if (
            not names
            or len(set(names)) != len(names)
            or any(not value for value in names)
            or refine.shape != (len(names),)
            or coarsen.shape != refine.shape
            or np.any(~np.isfinite(refine))
            or np.any(~np.isfinite(coarsen))
            or np.any(refine <= coarsen)
            or np.any(coarsen < 0.0)
        ):
            raise ValueError("High-enthalpy AMR names or thresholds are invalid.")
        self.indicator_names = names
        self.refine_thresholds = jnp.asarray(refine)
        self.coarsen_thresholds = jnp.asarray(coarsen)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "high-enthalpy-amr-indicators",
                "names": names,
                "refine": tuple(float(value) for value in refine),
                "coarsen": tuple(float(value) for value in coarsen),
            }
        )

    def evaluate(self, indicators: Mapping[str, ArrayLike], /) -> HighEnthalpyAMREvidence:
        if tuple(indicators) != self.indicator_names:
            raise ValueError(
                "AMR indicator mapping must follow the exact declared order."
            )
        values = jnp.stack(
            tuple(jnp.asarray(indicators[name]) for name in self.indicator_names), axis=-1
        )
        refine = jnp.any(values >= self.refine_thresholds, axis=-1)
        coarsen = jnp.all(values <= self.coarsen_thresholds, axis=-1)
        finite = jnp.all(jnp.isfinite(values), axis=-1)
        return HighEnthalpyAMREvidence(
            values,
            refine & finite,
            coarsen & finite,
            finite,
            jnp.all(finite),
            self.plan_id,
        )


class AerothermodynamicALEEvidence(StrictModule):
    candidate_vertices: Array
    volume_change: Array
    swept_volume: Array
    geometric_conservation_defect: Array
    minimum_volume: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class AerothermodynamicALEPlan(StrictModule, NonTrainableState):
    vertex_to_cell_volume: Array
    minimum_volume: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self, vertex_to_cell_volume: ArrayLike, /, *, minimum_volume: float = 1.0e-14
    ):
        projection = np.asarray(vertex_to_cell_volume, dtype=float)
        minimum = float(minimum_volume)
        if (
            projection.ndim != 2
            or np.any(~np.isfinite(projection))
            or not np.isfinite(minimum)
            or minimum <= 0.0
        ):
            raise ValueError("ALE volume projection or minimum volume is invalid.")
        self.vertex_to_cell_volume = jnp.asarray(projection)
        self.minimum_volume = minimum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "aerothermodynamic-ale",
                "shape": projection.shape,
                "minimum_volume": minimum,
            }
        )

    def evaluate(
        self,
        vertices: ArrayLike,
        vertex_displacement: ArrayLike,
        old_cell_volumes: ArrayLike,
        swept_face_volumes: ArrayLike,
        /,
    ) -> AerothermodynamicALEEvidence:
        points = jnp.asarray(vertices)
        displacement = jnp.asarray(vertex_displacement, dtype=points.dtype)
        old_volume = jnp.asarray(old_cell_volumes, dtype=points.dtype)
        swept = jnp.asarray(swept_face_volumes, dtype=points.dtype)
        if (
            displacement.shape != points.shape
            or self.vertex_to_cell_volume.shape[1] != points.shape[0]
            or old_volume.shape != (self.vertex_to_cell_volume.shape[0],)
            or swept.shape[0] != old_volume.size
        ):
            raise ValueError("ALE vertices, volumes, or swept fluxes are incompatible.")
        candidate = points + displacement
        displacement_magnitude = jnp.sqrt(jnp.sum(displacement * displacement, axis=-1))
        volume_change = self.vertex_to_cell_volume @ displacement_magnitude
        new_volume = old_volume + volume_change
        swept_total = jnp.sum(swept, axis=-1) if swept.ndim > 1 else swept
        defect = volume_change - swept_total
        finite = (
            jnp.all(jnp.isfinite(candidate))
            & jnp.all(jnp.isfinite(new_volume))
            & jnp.all(jnp.isfinite(defect))
        )
        scale = jnp.maximum(jnp.max(jnp.abs(volume_change)), 1.0)
        successful = (
            finite
            & jnp.all(new_volume >= self.minimum_volume)
            & jnp.all(jnp.abs(defect) <= 1.0e-10 * scale)
        )
        return AerothermodynamicALEEvidence(
            candidate,
            volume_change,
            swept_total,
            defect,
            jnp.min(new_volume),
            finite,
            successful,
            self.plan_id,
        )


class AerothermodynamicTopologyTransaction(StrictModule, NonTrainableState):
    old_topology_id: str = eqx.field(static=True)
    new_topology_id: str = eqx.field(static=True)
    remap: ConservativeRecessionRemapPlan
    transaction_id: str = eqx.field(static=True)

    def __init__(
        self,
        old_topology_id: str,
        new_topology_id: str,
        remap: ConservativeRecessionRemapPlan,
        /,
    ):
        old = str(old_topology_id)
        new = str(new_topology_id)
        if (
            not old
            or not new
            or old == new
            or not isinstance(remap, ConservativeRecessionRemapPlan)
        ):
            raise ValueError("Topology transaction identities or remap are invalid.")
        self.old_topology_id = old
        self.new_topology_id = new
        self.remap = remap
        self.transaction_id = canonical_fingerprint(
            {
                "kind": "aerothermodynamic-topology-transaction",
                "old": old,
                "new": new,
                "remap": remap.plan_id,
            }
        )


__all__ = [
    "AerothermodynamicALEEvidence",
    "AerothermodynamicALEPlan",
    "AerothermodynamicTopologyTransaction",
    "HighEnthalpyAMREvidence",
    "HighEnthalpyAMRIndicatorPlan",
]
