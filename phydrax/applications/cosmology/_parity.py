#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._closure import ScientificArtifactEnvelope


class ParityProfile(StrictModule, NonTrainableState):
    """Bounded physics, outputs, references, and error budgets for one parity claim."""

    name: str = eqx.field(static=True)
    equations: tuple[str, ...] = eqx.field(static=True)
    species: tuple[str, ...] = eqx.field(static=True)
    geometry: str = eqx.field(static=True)
    approximations: tuple[str, ...] = eqx.field(static=True)
    outputs: tuple[str, ...] = eqx.field(static=True)
    references: tuple[str, ...] = eqx.field(static=True)
    metrics: tuple[str, ...] = eqx.field(static=True)
    negative_boundaries: tuple[str, ...] = eqx.field(static=True)
    profile_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        name: str,
        equations: tuple[str, ...],
        species: tuple[str, ...],
        geometry: str,
        approximations: tuple[str, ...],
        outputs: tuple[str, ...],
        references: tuple[str, ...],
        metrics: tuple[str, ...],
        negative_boundaries: tuple[str, ...],
    ):
        name_ = str(name).strip()
        geometry_ = str(geometry).strip()
        groups = tuple(
            tuple(str(value).strip() for value in group)
            for group in (
                equations,
                species,
                approximations,
                outputs,
                references,
                metrics,
                negative_boundaries,
            )
        )
        if (
            not name_
            or not geometry_
            or any(not group or any(not value for value in group) for group in groups)
        ):
            raise ValueError("Parity profile fields must be non-empty.")
        self.name = name_
        self.equations = groups[0]
        self.species = groups[1]
        self.geometry = geometry_
        self.approximations = groups[2]
        self.outputs = groups[3]
        self.references = groups[4]
        self.metrics = groups[5]
        self.negative_boundaries = groups[6]
        self.profile_id = canonical_fingerprint(
            {
                "kind": "physics-parity-profile",
                "name": name_,
                "equations": list(groups[0]),
                "species": list(groups[1]),
                "geometry": geometry_,
                "approximations": list(groups[2]),
                "outputs": list(groups[3]),
                "references": list(groups[4]),
                "metrics": list(groups[5]),
                "negative_boundaries": list(groups[6]),
            }
        )


class ParityEvidence(StrictModule):
    metric_values: Array
    metric_limits: Array
    passed: Array
    finite: Array
    successful: Array
    profile_id: str = eqx.field(static=True)
    artifact: ScientificArtifactEnvelope

    def __init__(
        self,
        profile: ParityProfile,
        metric_values: ArrayLike,
        metric_limits: ArrayLike,
        artifact: ScientificArtifactEnvelope,
        /,
    ):
        values = jnp.asarray(metric_values)
        limits = jnp.asarray(metric_limits, dtype=values.dtype)
        if values.shape != (len(profile.metrics),) or limits.shape != values.shape:
            raise ValueError("Parity metric arrays must match the profile metric layout.")
        passed = values <= limits
        finite = jnp.all(jnp.isfinite(values)) & jnp.all(jnp.isfinite(limits))
        self.metric_values = values
        self.metric_limits = limits
        self.passed = passed
        self.finite = finite
        self.successful = finite & jnp.all(passed)
        self.profile_id = profile.profile_id
        self.artifact = artifact


__all__ = ["ParityEvidence", "ParityProfile"]
