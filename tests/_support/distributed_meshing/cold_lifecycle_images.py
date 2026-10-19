# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Categorical image construction and interpreted-source restoration records."""

from typing import Any

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.geometry.multiregion_surface._label_extraction import LabelFieldVolumeBinding
from phydrax.lifecycle._meshing_sources import read_meshing_source_closure
from tests._support.distributed_meshing.cold_lifecycle_identities import (
    image_source_identity,
)


def categorical_extraction() -> (
    phx.geometry.multiregion_surface.LabelFieldSurfaceExtractionResult
):
    labels = np.zeros((2, 2, 2), dtype=np.int32)
    labels[1] = 1
    state = phx.threshold_dynamics.LabelFieldState(
        jnp.asarray(labels, dtype=jnp.int32),
        jnp.asarray((True, True), dtype=jnp.bool_),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0, dtype=jnp.float64),
        label_ids=("left", "right"),
        route_id="categorical-image-test",
        prepared_id="categorical-image-prepared",
        site_id="categorical-image-sites",
    )
    capacity = phx.geometry.multiregion_surface.MultiRegionSurfaceCapacityPlan(
        vertex_capacity=4096,
        edge_capacity=8192,
        face_capacity=8192,
        region_capacity=4,
        region_pair_capacity=6,
        maximum_edge_valence=3,
        maximum_vertex_region_pairs=6,
        resource_id="categorical-image-volume",
    )
    return (
        phx.geometry.multiregion_surface.LabelFieldSurfaceExtractionPlan(
            ("left", "right"), capacity, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)
        )
        .prepare(labels.shape)
        .extract(state)
    )


def interpreted_image_source(path: str, content_id: str) -> dict[str, Any]:
    source = read_meshing_source_closure(path, expected_content_id=content_id)
    if not isinstance(
        source, (phx.geometry.CompartmentMeshingSource, LabelFieldVolumeBinding)
    ):
        raise TypeError(
            f"cold_lifecycle.image_source must be an interpreted image source; got {type(source).__name__}"
        )
    return image_source_identity(source)
