#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Core-local to composite material identity for immutable layer volumes."""

from __future__ import annotations

import numpy as np
from jax.typing import ArrayLike

import phydrax.meshing._plc_source as plc_source_owner

from .._validation import unique_identifiers
from ..typing import checked


@checked
def prepare_layer_region_identity(
    complex_: plc_source_owner.PiecewiseLinearComplex,
    layer_regions: np.ndarray,
    region_ids: tuple[str, ...] | None,
    core_region_map: ArrayLike | None,
    /,
) -> tuple[tuple[str, ...], np.ndarray]:
    """Validate the explicit material namespace shared by layers and PLC core."""
    if (region_ids is None) != (core_region_map is None):
        raise ValueError(
            "NativeLayerCoreSource.core_region_map must provide its region namespace and core-local map together"
        )
    names = (
        complex_.region_ids
        if region_ids is None
        else unique_identifiers(region_ids, "region_ids")
    )
    mapping = (
        np.arange(complex_.region_count, dtype=np.int64)
        if core_region_map is None
        else np.asarray(core_region_map)
    )
    if not np.issubdtype(mapping.dtype, np.integer) or mapping.shape != (
        complex_.region_count,
    ):
        raise TypeError(
            "NativeLayerCoreSource.core_region_map must contain one integer composite material index per core region"
        )
    if (
        np.any(mapping < 0)
        or np.any(mapping >= len(names))
        or np.unique(mapping).size != mapping.size
    ):
        raise ValueError(
            "NativeLayerCoreSource.core_region_map must map injectively into the declared composite namespace"
        )
    if any(
        names[int(index)] != name
        for index, name in zip(mapping, complex_.region_ids, strict=True)
    ):
        raise ValueError(
            "NativeLayerCoreSource.core_region_map must retain the declared material names"
        )
    if np.any(layer_regions < 0) or np.any(layer_regions >= len(names)):
        raise ValueError(
            "NativeLayerCoreSource.layer_regions must index the declared composite material namespace"
        )
    occupied = np.unique(
        np.concatenate(
            (
                layer_regions,
                mapping[np.unique(complex_.facet_regions[complex_.facet_regions >= 0])],
            )
        )
    )
    if not np.array_equal(occupied, np.arange(len(names), dtype=np.int64)):
        raise ValueError(
            "NativeLayerCoreSource.core_region_map must have realized layer or bounded core material incidence"
        )
    result = np.array(mapping, dtype=np.int64, copy=True)
    result.setflags(write=False)
    return names, result


__all__ = ["prepare_layer_region_identity"]
