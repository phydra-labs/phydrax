#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact core-local to composite material identity for immutable layer volumes."""

from __future__ import annotations

import numpy as np
from jax.typing import ArrayLike

from .._validation import unique_identifiers
from ._volume_generation import PiecewiseLinearComplex


def prepare_layer_region_identity(
    complex_: PiecewiseLinearComplex,
    layer_regions: np.ndarray,
    region_ids: tuple[str, ...] | None,
    core_region_map: ArrayLike | None,
    /,
) -> tuple[tuple[str, ...], np.ndarray]:
    if (region_ids is None) != (core_region_map is None):
        raise ValueError(
            "Composite material identity requires its region namespace and exact core-local map together."
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
            "core_region_map must contain one integer composite material index per core region."
        )
    if (
        np.any(mapping < 0)
        or np.any(mapping >= len(names))
        or np.unique(mapping).size != mapping.size
    ):
        raise ValueError(
            "Core material identities must map injectively into the declared composite namespace."
        )
    if any(
        names[int(index)] != name
        for index, name in zip(mapping, complex_.region_ids, strict=True)
    ):
        raise ValueError(
            "Core-local and composite maps must retain the exact authoritative material names."
        )
    if np.any(layer_regions < 0) or np.any(layer_regions >= len(names)):
        raise ValueError(
            "Every layer cell requires an authoritative composite material index."
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
            "Every composite region must have actual layer or bounded core material incidence."
        )
    mapping = np.array(mapping, dtype=np.int64, copy=True)
    mapping.setflags(write=False)
    return names, mapping


__all__ = ["prepare_layer_region_identity"]
