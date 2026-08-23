#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._precision import PrecisionEvidenceEnvelope
from .._strict import StrictModule
from ..metrix import AtlasCover
from ._precision import IntegrationPrecisionPolicy


class AtlasPatchQuadrature(StrictModule):
    """Fixed quadrature design and overlap weight in one chart."""

    coordinates: Array
    weights: Array
    ownership_weights: Array
    chart_index: int
    patch_id: str

    def __init__(
        self,
        chart_index: int,
        coordinates: ArrayLike,
        weights: ArrayLike,
        /,
        *,
        ownership_weights: ArrayLike | None = None,
        patch_id: str,
    ):
        points = jnp.asarray(coordinates)
        weights_ = jnp.asarray(weights, dtype=points.real.dtype)
        if points.ndim != 2 or weights_.shape != (points.shape[0],):
            raise ValueError(
                "Atlas patch quadrature requires (samples, dim) points and weights."
            )
        ownership = (
            jnp.ones_like(weights_)
            if ownership_weights is None
            else jnp.asarray(ownership_weights, dtype=weights_.dtype)
        )
        if ownership.shape != weights_.shape:
            raise ValueError("ownership_weights must match quadrature weights.")
        identifier = str(patch_id)
        if not identifier:
            raise ValueError("patch_id must be non-empty.")
        self.chart_index = int(chart_index)
        self.coordinates = points
        self.weights = weights_
        self.ownership_weights = ownership
        self.patch_id = identifier


class AtlasIntegrationTarget(StrictModule):
    """Explicit patch quadratures over one fixed atlas cover."""

    cover: AtlasCover
    patches: tuple[AtlasPatchQuadrature, ...]
    target_id: str

    def __init__(
        self,
        cover: AtlasCover,
        patches: Sequence[AtlasPatchQuadrature],
        /,
        *,
        target_id: str,
    ):
        if not isinstance(cover, AtlasCover):
            raise TypeError("cover must be an AtlasCover.")
        patches_ = tuple(patches)
        if not patches_:
            raise ValueError("Atlas integration requires at least one patch.")
        for patch in patches_:
            if not isinstance(patch, AtlasPatchQuadrature):
                raise TypeError("patches must contain AtlasPatchQuadrature objects.")
            if not (0 <= patch.chart_index < len(cover.atlas.charts)):
                raise ValueError("Patch chart index is outside the atlas.")
            chart = cover.atlas.charts[patch.chart_index]
            if patch.coordinates.shape[-1] != chart.dimension:
                raise ValueError("Patch quadrature dimension does not match its chart.")
        identifier = str(target_id)
        if not identifier:
            raise ValueError("target_id must be non-empty.")
        self.cover = cover
        self.patches = patches_
        self.target_id = identifier


class AtlasIntegrationResult(StrictModule):
    value: Array
    patch_values: Array
    represented_weight: Array
    valid: Array
    target_id: str
    precision_evidence: PrecisionEvidenceEnvelope

    def __init__(
        self,
        value: ArrayLike,
        patch_values: ArrayLike,
        represented_weight: ArrayLike,
        /,
        *,
        valid: ArrayLike,
        target_id: str,
        precision_evidence: PrecisionEvidenceEnvelope,
    ):
        self.value = jnp.asarray(value)
        self.patch_values = jnp.asarray(patch_values)
        self.represented_weight = jnp.asarray(represented_weight)
        self.valid = jnp.asarray(valid, dtype=bool)
        if not isinstance(precision_evidence, PrecisionEvidenceEnvelope):
            raise TypeError("precision_evidence must be PrecisionEvidenceEnvelope.")
        self.precision_evidence = precision_evidence
        self.target_id = str(target_id)


def integrate_atlas_scalar(
    target: AtlasIntegrationTarget,
    local_fields: Sequence[Callable[[Array], Array]],
    /,
    *,
    precision: IntegrationPrecisionPolicy | None = None,
) -> AtlasIntegrationResult:
    """Integrate scalar local representatives with explicit overlap weights."""
    if not isinstance(target, AtlasIntegrationTarget):
        raise TypeError("target must be an AtlasIntegrationTarget.")
    fields = tuple(local_fields)
    if len(fields) != len(target.cover.atlas.charts) or any(
        not callable(field) for field in fields
    ):
        raise ValueError("One callable local field is required per atlas chart.")
    precision_ = IntegrationPrecisionPolicy() if precision is None else precision
    if not isinstance(precision_, IntegrationPrecisionPolicy):
        raise TypeError("precision must be an IntegrationPrecisionPolicy or None.")
    contributions = []
    represented = precision_.accumulation(0.0)
    valid = jnp.asarray(True)
    for patch in target.patches:
        coordinates = precision_.evaluation(patch.coordinates)
        support = target.cover.support(patch.chart_index, coordinates)
        values = precision_.evaluation(fields[patch.chart_index](coordinates))
        if values.shape != (patch.coordinates.shape[0],):
            raise ValueError("Atlas scalar field must return one value per sample.")
        effective = precision_.accumulation(patch.weights * patch.ownership_weights)
        contributions.append(
            jnp.sum(precision_.accumulation(jnp.where(support, effective * values, 0.0)))
        )
        represented = precision_.accumulation(
            represented
            + jnp.sum(precision_.accumulation(jnp.where(support, effective, 0.0)))
        )
        valid = (
            valid
            & jnp.all(jnp.isfinite(values))
            & jnp.all(jnp.isfinite(effective) & (effective >= 0.0))
            & jnp.all((patch.ownership_weights >= 0.0) & (patch.ownership_weights <= 1.0))
        )
    patch_values = precision_.accumulation(jnp.stack(contributions))
    accumulated_value = jnp.sum(patch_values)
    value = precision_.output(accumulated_value)
    represented = precision_.decision(represented)
    valid = valid & jnp.isfinite(accumulated_value) & (represented > 0.0)
    return AtlasIntegrationResult(
        value,
        patch_values,
        represented,
        valid=valid,
        precision_evidence=precision_.evidence_for(patch_values),
        target_id=target.target_id,
    )


__all__ = [
    "AtlasIntegrationResult",
    "AtlasIntegrationTarget",
    "AtlasPatchQuadrature",
    "integrate_atlas_scalar",
]
