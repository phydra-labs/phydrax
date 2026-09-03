#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._moving_conservation import MovingTraceRoute


class SlidingMortarPlan(StrictModule, NonTrainableState):
    routes: tuple[MovingTraceRoute, ...]
    overlap_fractions: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        routes: Sequence[MovingTraceRoute],
        overlap_fractions: ArrayLike,
        /,
    ):
        routes_ = tuple(routes)
        overlap = jnp.asarray(overlap_fractions)
        if (
            not routes_
            or any(not isinstance(value, MovingTraceRoute) for value in routes_)
            or overlap.shape != (len(routes_),)
            or bool(jnp.any((overlap < 0.0) | (overlap > 1.0)))
        ):
            raise ValueError("Sliding mortar routes or overlap fractions are invalid.")
        self.routes = routes_
        self.overlap_fractions = overlap
        self.plan_id = canonical_fingerprint(
            {
                "kind": "sliding-mortar-plan",
                "routes": tuple(value.route_id for value in routes_),
                "overlap": array_tree_fingerprint(np.asarray(overlap)),
            }
        )

    def flux_contributions(
        self,
        fraction: ArrayLike,
        fluxes: Sequence[ArrayLike],
        /,
    ) -> tuple[tuple[Array, Array], ...]:
        if len(fluxes) != len(self.routes):
            raise ValueError("Sliding mortar flux count changed.")
        contributions = []
        for route, overlap, flux in zip(
            self.routes, self.overlap_fractions, fluxes, strict=True
        ):
            prepared = route.at(fraction)
            value = jnp.asarray(flux) * overlap
            if prepared.mortar is not None:
                contributions.append(
                    prepared.mortar.conservative_flux_contributions(value)
                )
            else:
                owner = ein.contract(
                    "q,qi,qv->iv",
                    prepared.physical_weights,
                    prepared.owner_basis,
                    value,
                    backend="jax",
                )
                neighbour = -ein.contract(
                    "q,qi,qv->iv",
                    prepared.physical_weights,
                    prepared.neighbour_basis,
                    value,
                    backend="jax",
                )
                contributions.append((owner, neighbour))
        return tuple(contributions)


class CutCellConservationPlan(StrictModule, NonTrainableState):
    volume_fractions: Array
    face_apertures: Array
    merge_targets: Array
    active: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        volume_fractions: ArrayLike,
        face_apertures: ArrayLike,
        merge_targets: ArrayLike,
        /,
        *,
        minimum_volume_fraction: float = 0.05,
    ):
        volumes = jnp.asarray(volume_fractions)
        apertures = jnp.asarray(face_apertures)
        targets = jnp.asarray(merge_targets, dtype=jnp.int32)
        if (
            volumes.ndim != 1
            or apertures.ndim != 2
            or apertures.shape[0] != volumes.shape[0]
            or targets.shape != volumes.shape
            or bool(jnp.any((volumes < 0.0) | (volumes > 1.0)))
            or bool(jnp.any((apertures < 0.0) | (apertures > 1.0)))
        ):
            raise ValueError("Cut-cell fractions, apertures, or merge targets invalid.")
        small = (volumes > 0.0) & (volumes < float(minimum_volume_fraction))
        if bool(jnp.any(small & ((targets < 0) | (targets >= volumes.size)))):
            raise ValueError("Every small cut cell requires a valid merge target.")
        self.volume_fractions = volumes
        self.face_apertures = apertures
        self.merge_targets = targets
        self.active = volumes > 0.0
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cut-cell-conservation-plan",
                "volumes": array_tree_fingerprint(np.asarray(volumes)),
                "apertures": array_tree_fingerprint(np.asarray(apertures)),
                "targets": array_tree_fingerprint(np.asarray(targets)),
            }
        )

    def merge_small_cell_contents(self, contents: ArrayLike, /) -> Array:
        values = jnp.asarray(contents)
        if values.shape[0] != self.volume_fractions.shape[0]:
            raise ValueError("Cut-cell contents changed cell count.")
        result = values
        for cell in range(self.volume_fractions.shape[0]):
            small = (self.volume_fractions[cell] > 0.0) & (
                self.volume_fractions[cell] < 0.05
            )
            target = jnp.maximum(self.merge_targets[cell], 0)
            result = result.at[target].add(jnp.where(small, result[cell], 0.0))
            result = result.at[cell].set(jnp.where(small, 0.0, result[cell]))
        return result


class OversetConnectivity(StrictModule, NonTrainableState):
    donor_cells: Array
    receptor_cells: Array
    interpolation_weights: Array
    donor_content_weights: Array
    receptor_content_weights: Array
    active: Array
    connectivity_id: str = eqx.field(static=True)

    def __init__(
        self,
        donor_cells: ArrayLike,
        receptor_cells: ArrayLike,
        interpolation_weights: ArrayLike,
        donor_content_weights: ArrayLike,
        receptor_content_weights: ArrayLike,
        active: ArrayLike,
        /,
    ):
        donors = np.asarray(donor_cells, dtype=np.int32)
        receptors = np.asarray(receptor_cells, dtype=np.int32)
        interpolation = np.asarray(interpolation_weights, dtype=float)
        donor_weights = np.asarray(donor_content_weights, dtype=float)
        receptor_weights = np.asarray(receptor_content_weights, dtype=float)
        active_ = np.asarray(active, dtype=bool)
        count = receptors.shape[0]
        if (
            donors.ndim != 2
            or receptors.ndim != 1
            or interpolation.shape != donors.shape
            or donor_weights.shape != donors.shape
            or receptor_weights.shape != (count,)
            or active_.shape != (count,)
            or np.any(interpolation < 0.0)
            or np.any(donor_weights < 0.0)
            or np.any(receptor_weights < 0.0)
            or np.max(np.abs(np.sum(interpolation, axis=1) - 1.0)) > 1.0e-10
        ):
            raise ValueError("Overset connectivity weights or shapes are invalid.")
        self.donor_cells = jnp.asarray(donors)
        self.receptor_cells = jnp.asarray(receptors)
        self.interpolation_weights = jnp.asarray(interpolation)
        self.donor_content_weights = jnp.asarray(donor_weights)
        self.receptor_content_weights = jnp.asarray(receptor_weights)
        self.active = jnp.asarray(active_)
        self.connectivity_id = canonical_fingerprint(
            {
                "kind": "overset-connectivity",
                "donors": array_tree_fingerprint(donors),
                "receptors": array_tree_fingerprint(receptors),
                "interpolation": array_tree_fingerprint(interpolation),
                "donor_content": array_tree_fingerprint(donor_weights),
                "receptor_content": array_tree_fingerprint(receptor_weights),
                "active": array_tree_fingerprint(active_),
            }
        )


class OversetTransferResult(StrictModule, NonTrainableState):
    receptor_state: Array
    donor_content_correction: Array
    receptor_content: Array
    conservation_defect: Array
    successful: Array
    transfer_id: str = eqx.field(static=True)


class ConservativeOversetPlan(StrictModule, NonTrainableState):
    connectivity: OversetConnectivity
    plan_id: str = eqx.field(static=True)

    def __init__(self, connectivity: OversetConnectivity, /):
        if not isinstance(connectivity, OversetConnectivity):
            raise TypeError("connectivity must be OversetConnectivity.")
        self.connectivity = connectivity
        self.plan_id = canonical_fingerprint(
            {
                "kind": "conservative-overset-plan",
                "connectivity": connectivity.connectivity_id,
            }
        )

    def transfer(self, donor_state: ArrayLike, /) -> OversetTransferResult:
        state = jnp.asarray(donor_state)
        donor_values = state[self.connectivity.donor_cells]
        receptor_state = ein.contract(
            "rk,rkv->rv",
            self.connectivity.interpolation_weights,
            donor_values,
            backend="jax",
        )
        receptor_content = (
            self.connectivity.receptor_content_weights[:, None] * receptor_state
        )
        donor_content = ein.contract(
            "rk,rkv->rv",
            self.connectivity.donor_content_weights,
            donor_values,
            backend="jax",
        )
        correction_per_route = donor_content - receptor_content
        correction = jnp.zeros_like(state)
        normalization = jnp.sum(self.connectivity.donor_content_weights, axis=1)
        normalized = self.connectivity.donor_content_weights / jnp.maximum(
            normalization[:, None], 1.0e-30
        )
        for route in range(self.connectivity.donor_cells.shape[0]):
            cells = self.connectivity.donor_cells[route]
            local = normalized[route, :, None] * correction_per_route[route]
            correction = correction.at[cells].add(local)
        defect = jnp.sum(donor_content - correction_per_route - receptor_content, axis=0)
        maximum_defect = jnp.max(jnp.abs(defect))
        transfer_id = canonical_fingerprint(
            {
                "kind": "conservative-overset-transfer",
                "plan": self.plan_id,
                "state_shape": tuple(state.shape),
            }
        )
        return OversetTransferResult(
            receptor_state,
            correction,
            receptor_content,
            maximum_defect,
            maximum_defect <= 1.0e-10,
            transfer_id,
        )


__all__ = [
    "ConservativeOversetPlan",
    "CutCellConservationPlan",
    "OversetConnectivity",
    "OversetTransferResult",
    "SlidingMortarPlan",
]
