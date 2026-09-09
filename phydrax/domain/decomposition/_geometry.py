#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.random as jr
from jaxtyping import Array, Key

from ..._doc import DOC_KEY0
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._structure import PointSampling
from ._cover import (
    PairedSupportEvidence,
    SubdomainCover,
    SubdomainCoverEvidence,
)


class MappedCoverValidationPlan(StrictModule, NonTrainableState):
    """Fixed sampled audit policy for a user-authored mapped cover."""

    ambient_points: int = eqx.field(static=True)
    pairing_points: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    sampler: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        ambient_points: int = 1024,
        pairing_points: int = 256,
        tolerance: float = 1.0e-8,
        sampler: str = "latin_hypercube",
    ):
        ambient = int(ambient_points)
        pairing = int(pairing_points)
        tolerance_ = float(tolerance)
        if ambient <= 0 or pairing <= 0:
            raise ValueError("Mapped-cover audit point counts must be positive.")
        if not math.isfinite(tolerance_) or tolerance_ < 0.0:
            raise ValueError("Mapped-cover tolerance must be finite and non-negative.")
        if not isinstance(sampler, str) or not sampler:
            raise ValueError("sampler must be a non-empty string.")
        self.ambient_points = ambient
        self.pairing_points = pairing
        self.tolerance = tolerance_
        self.sampler = sampler


class MappedCoverEvidence(StrictModule, NonTrainableState):
    """Coverage and paired-map evidence for one mapped cover."""

    coverage: SubdomainCoverEvidence
    pairings: tuple[PairedSupportEvidence, ...]
    verified: bool = eqx.field(static=True)

    def __init__(
        self,
        coverage: SubdomainCoverEvidence,
        pairings: tuple[PairedSupportEvidence, ...],
        /,
    ):
        self.coverage = coverage
        self.pairings = tuple(pairings)
        self.verified = coverage.verified and all(value.verified for value in pairings)


def validate_mapped_cover(
    cover: SubdomainCover,
    plan: MappedCoverValidationPlan | None = None,
    /,
    *,
    key: Key[Array, ""] = DOC_KEY0,
    raise_on_error: bool = True,
) -> MappedCoverEvidence:
    """Sample cover support and every paired physical-coordinate realization."""
    if not isinstance(cover, SubdomainCover):
        raise TypeError("cover must be a SubdomainCover.")
    plan_ = MappedCoverValidationPlan() if plan is None else plan
    if not isinstance(plan_, MappedCoverValidationPlan):
        raise TypeError("plan must be a MappedCoverValidationPlan or None.")
    keys = jr.split(key, len(cover.pairings) + 1)
    ambient_batch = cover.ambient.component().sample(
        PointSampling(plan_.ambient_points, design=plan_.sampler),
        key=keys[0],
    )
    coverage = cover.audit(ambient_batch)
    pairing_evidence = []
    for index, pairing in enumerate(cover.pairings):
        batch = pairing.component.sample(
            PointSampling(plan_.pairing_points, design=plan_.sampler),
            key=keys[index + 1],
        )
        pairing_evidence.append(
            pairing.audit(
                batch,
                cover.patch(pairing.left_patch_id),
                cover.patch(pairing.right_patch_id),
                tolerance=plan_.tolerance,
            )
        )
    evidence = MappedCoverEvidence(coverage, tuple(pairing_evidence))
    if raise_on_error and not evidence.verified:
        raise ValueError(
            "Mapped subdomain cover validation failed: "
            f"coverage={coverage.verified}, "
            f"invalid_pairings={tuple(value.pairing_id for value in pairing_evidence if not value.verified)}."
        )
    return evidence


__all__ = [
    "MappedCoverEvidence",
    "MappedCoverValidationPlan",
    "validate_mapped_cover",
]
