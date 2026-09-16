#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ._scft import PreparedSCFT


class SCFTSymmetryEvidence(StrictModule):
    original_defects: Array
    projected_defects: Array
    projection_distance: Array
    projected_fields: Array
    successful: Array
    prepared_id: str = eqx.field(static=True)


def scft_symmetry_evidence(
    prepared: PreparedSCFT,
    fields: ArrayLike,
    /,
    *,
    tolerance: float = 1.0e-9,
) -> SCFTSymmetryEvidence:
    if not isinstance(prepared, PreparedSCFT):
        raise TypeError("prepared must be PreparedSCFT.")
    threshold = float(tolerance)
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("tolerance must be finite and non-negative.")
    if not prepared.symmetries:
        raise ValueError("SCFT symmetry evidence requires declared finite-group actions.")
    value = jnp.asarray(fields)
    if value.shape != prepared.field_shape:
        raise ValueError(f"fields must have shape {prepared.field_shape}.")
    original = prepared.symmetry_defects(value)
    projected = prepared.project_initial_fields(value)
    projected_defects = prepared.symmetry_defects(projected)
    distance = jnp.linalg.norm((projected - value).reshape((-1,)))
    successful = (
        jnp.all(jnp.isfinite(original))
        & jnp.all(jnp.isfinite(projected_defects))
        & jnp.isfinite(distance)
        & jnp.all(projected_defects <= threshold)
    )
    return SCFTSymmetryEvidence(
        original,
        projected_defects,
        distance,
        projected,
        successful,
        prepared.prepared_id,
    )


__all__ = ["SCFTSymmetryEvidence", "scft_symmetry_evidence"]
