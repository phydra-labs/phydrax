#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Positive empirical cubature over one native finite integration measure."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

import phydrax.ein as ein

from .._measure_weights import normalized_weights
from .._strict import StrictModule
from ..coresets import CoresetSelection, moment_recombine, MomentRecombination
from ._measure_transform import (
    feature_matrix,
    lower_finite_measure,
    transformed_weighted_realization,
)


class EmpiricalCubaturePlan(StrictModule):
    """Positive moment recombination policy for a supplied finite feature span."""

    method: MomentRecombination

    def __init__(self, method: MomentRecombination | None = None, /):
        resolved = MomentRecombination() if method is None else method
        if not isinstance(resolved, MomentRecombination):
            raise TypeError("method must be a MomentRecombination or None.")
        self.method = resolved


class EmpiricalCubatureDiagnostics(StrictModule):
    """Complete normalized/physical moment and selector evidence."""

    moment_residual: Array
    relative_moment_residual: Array
    maximum_moment_residual: Array
    source_mass: Array
    selected_mass: Array
    valid: Array
    selection: CoresetSelection
    numerical_rank: Array
    active_support: Array
    feature_count: int = eqx.field(static=True)
    exactness: str = eqx.field(static=True)


def empirical_cubature(
    realization,
    basis_values,
    plan: EmpiricalCubaturePlan,
    /,
):
    """Recombine one positive finite measure while preserving supplied moments."""
    if not isinstance(plan, EmpiricalCubaturePlan):
        raise TypeError("plan must be an EmpiricalCubaturePlan.")
    measure = lower_finite_measure(realization)
    features = feature_matrix(basis_values, measure.axis, measure.count)
    if jnp.issubdtype(features.dtype, jnp.complexfloating):
        selector_features = jnp.concatenate(
            (features.real, features.imag), axis=1
        ).astype(features.real.dtype)
    else:
        selector_features = features
    selection = moment_recombine(
        selector_features,
        plan.method,
        log_weights=measure.log_weights,
        mask=measure.mask,
    )
    source_weights, _, source_valid, _ = normalized_weights(
        measure.count,
        log_weights=measure.log_weights,
        mask=measure.mask,
    )
    source_moment = ein.contract("i,ij->j", source_weights, selector_features)
    selected_weights = selection.weights
    selected_features = selector_features[selection.indices]
    selected_moment = ein.contract("i,ij->j", selected_weights, selected_features)
    residual = selected_moment - source_moment
    scale = jnp.maximum(
        jnp.linalg.vector_norm(source_moment),
        jnp.asarray(jnp.finfo(selector_features.dtype).tiny),
    )
    relative = jnp.linalg.vector_norm(residual) / scale
    maximum = jnp.max(jnp.abs(residual), initial=0.0)
    source_mass = measure.physical_mass
    selected_mass = source_mass * jnp.sum(selected_weights)
    valid = (
        source_valid
        & selection.diagnostics.valid
        & jnp.all(jnp.isfinite(residual))
        & jnp.isfinite(relative)
    )
    diagnostics = EmpiricalCubatureDiagnostics(
        moment_residual=residual,
        relative_moment_residual=relative,
        maximum_moment_residual=maximum,
        source_mass=source_mass,
        selected_mass=selected_mass,
        valid=valid,
        selection=selection,
        numerical_rank=selection.diagnostics.numerical_rank,
        active_support=selection.active_points,
        feature_count=int(selector_features.shape[1]),
        exactness="supplied-finite-feature-span-only",
    )
    compressed = transformed_weighted_realization(
        realization,
        measure,
        selection.log_weights,
        transformation_kind="empirical-cubature",
        transformation_diagnostics=diagnostics,
        provenance=f"empirical-cubature:{measure.source_provenance}",
        indices=selection.indices,
        selection_mask=selection.mask,
    )
    return compressed


__all__ = [
    "EmpiricalCubatureDiagnostics",
    "EmpiricalCubaturePlan",
    "empirical_cubature",
]
