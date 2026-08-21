#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
from jaxtyping import Array

from phydrax.coresets import (
    CoresetSelection,
    kernel_herd,
    KernelHerding,
    moment_recombine,
    MomentRecombination,
)

from .._strict import StrictModule
from ._measure_transform import (
    feature_matrix,
    lower_finite_measure,
    transformed_weighted_realization,
)


CompressionMethod = MomentRecombination | KernelHerding
FeatureMap = Callable[[Any], Any]


class MeasureCompressionDiagnostics(StrictModule):
    """Selection evidence and source identity for one compressed realization."""

    selection: Any
    source_mass: Array
    source_points: int = eqx.field(static=True)
    feature_count: int = eqx.field(static=True)
    source_provenance: str = eqx.field(static=True)


def _select(
    feature_values: Array,
    method: CompressionMethod,
    /,
    *,
    log_weights: Array,
    mask: Array | None,
) -> CoresetSelection:
    if isinstance(method, MomentRecombination):
        return moment_recombine(
            feature_values,
            method,
            log_weights=log_weights,
            mask=mask,
        )
    if isinstance(method, KernelHerding):
        return kernel_herd(
            feature_values,
            method,
            log_weights=log_weights,
            mask=mask,
        )
    raise TypeError("method must be MomentRecombination or KernelHerding.")


def compress(
    realization: Any,
    method: CompressionMethod,
    /,
    *,
    features: FeatureMap | Any | None = None,
):
    """Compress a finite positive realization before evaluating its integrand."""

    measure = lower_finite_measure(realization)
    raw_features = (
        measure.samples
        if features is None
        else features(measure.samples)
        if callable(features)
        else features
    )
    feature_values = feature_matrix(raw_features, measure.axis, measure.count)
    selection = _select(
        feature_values,
        method,
        log_weights=measure.log_weights,
        mask=measure.mask,
    )
    if not bool(selection.diagnostics.valid):
        raise ValueError("Coreset selection failed for the supplied measure.")
    diagnostics = MeasureCompressionDiagnostics(
        selection=selection.diagnostics,
        source_mass=measure.physical_mass,
        source_points=measure.count,
        feature_count=int(feature_values.shape[1]),
        source_provenance=measure.source_provenance,
    )
    provenance = f"compressed:{selection.method}:{measure.source_provenance}"
    return transformed_weighted_realization(
        realization,
        measure,
        selection.log_weights,
        transformation_kind="compression",
        transformation_diagnostics=diagnostics,
        provenance=provenance,
        indices=selection.indices,
    )


__all__ = [
    "CompressionMethod",
    "MeasureCompressionDiagnostics",
    "compress",
]
