#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Seed an explicit three-cell foam from a threshold-dynamics label field."""

from typing import Any

import jax.numpy as jnp
import numpy as np

import phydrax.threshold_dynamics as td
from phydrax.applications.foams import (
    FoamEquilibriumPlan,
    FoamMaterialPlan,
    PreparedFoamEquilibrium,
)
from phydrax.geometry.multiregion_surface import (
    MultiRegionSurfaceCapacityPlan,
    MultiRegionSurfaceValidationPolicy,
)


RESOLUTION = 3


def _labels() -> np.ndarray:
    spacing = 1.0 / RESOLUTION
    axis = (np.arange(RESOLUTION) + 0.5) * spacing
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    centers = np.asarray(((0.25, 0.25, 0.5), (0.75, 0.25, 0.5), (0.5, 0.78, 0.5)))
    return np.argmin(
        np.sum((points[..., None, :] - centers) ** 2, axis=-1), axis=-1
    ).astype(np.int32)


def run() -> dict[str, Any]:
    labels = _labels()
    scale = RESOLUTION**3
    capacity = MultiRegionSurfaceCapacityPlan(
        vertex_capacity=48 * scale,
        edge_capacity=144 * scale,
        face_capacity=96 * scale,
        region_capacity=4,
        region_pair_capacity=6,
        maximum_edge_valence=3,
        maximum_vertex_region_pairs=6,
        resource_id="threshold-surface-example",
    )
    spacing = 1.0 / RESOLUTION
    label_ids = ("cell-a", "cell-b", "cell-c")
    extraction = td.LabelFieldSurfaceExtractionPlan(
        label_ids,
        capacity,
        spacing=(spacing, spacing, spacing),
        origin=(0.5 * spacing, 0.5 * spacing, 0.5 * spacing),
        validation_policy=MultiRegionSurfaceValidationPolicy(profile="dry_foam"),
    ).prepare(labels.shape)
    counts = np.bincount(labels.reshape(-1), minlength=len(label_ids))
    threshold_state = td.LabelFieldState(
        jnp.asarray(labels),
        jnp.asarray(counts > 0),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0, dtype=jnp.float64),
        label_ids=label_ids,
        route_id="uniform-grid-label-seed",
        prepared_id=extraction.prepared_id,
        site_id=extraction.prepared_id,
    )
    result = extraction.extract(threshold_state)
    if not result.accepted or result.topology is None or result.state is None:
        raise RuntimeError(
            "Threshold-surface extraction failed with status "
            f"{result.evidence.status.name}."
        )
    if result.surface is None:
        raise RuntimeError("Accepted extraction did not provide a prepared surface.")
    material = FoamMaterialPlan.soap_film(result.topology.region_ids, 0.025)
    equilibrium = PreparedFoamEquilibrium(
        FoamEquilibriumPlan(), result.surface, material, result.state
    )
    validation = result.evidence.validation
    if validation is None:
        raise RuntimeError("Accepted extraction did not provide validation evidence.")
    return {
        "source_epoch": result.lineage.source_epoch,
        "source_binding_id": result.lineage.source_binding_id,
        "region_ids": result.topology.region_ids,
        "vertices": result.topology.vertex_count,
        "faces": result.topology.face_count,
        "maximum_edge_valence": validation.maximum_edge_valence,
        "ambiguous_cells_resolved": result.evidence.ambiguous_cell_count,
        "maximum_volume_relative_error": max(
            result.evidence.finite_region_volume_relative_errors
        ),
        "maximum_area_relative_error": max(result.evidence.pair_area_relative_errors),
        "collision_certified": result.evidence.collision_certified,
        "equilibrium_route": equilibrium.route,
        "authority": "explicit-multiregion-surface",
    }


if __name__ == "__main__":
    print(run())
