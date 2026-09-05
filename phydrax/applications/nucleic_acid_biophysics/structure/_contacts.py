# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ...._strict import StrictModule
from ....ein import contract
from .._pair_graph import BaseInteraction, BaseInteractionGraph
from ._ermsd import NucleotideGDescriptor
from ._frames import base_frames


@dataclass(frozen=True, slots=True)
class GeometricContactCriteria:
    """Caller-declared geometry criteria in angstrom; no hydrogen-bond claim."""

    name: str
    maximum_distance: float
    minimum_abs_normal_cosine: float
    coplanar_height: float
    stacking_height: tuple[float, float]
    stacking_lateral_radius: float

    def __post_init__(self):
        numbers = (
            self.maximum_distance,
            self.minimum_abs_normal_cosine,
            self.coplanar_height,
            *self.stacking_height,
            self.stacking_lateral_radius,
        )
        if (
            not self.name
            or not all(np.isfinite(x) for x in numbers)
            or self.maximum_distance <= 0
            or not 0 <= self.minimum_abs_normal_cosine <= 1
            or not 0
            <= self.coplanar_height
            < self.stacking_height[0]
            < self.stacking_height[1]
            or self.stacking_lateral_radius <= 0
        ):
            raise ValueError(
                "Contact criteria require explicit ordered positive geometry thresholds."
            )


class GeometricContactEvaluation(StrictModule):
    coplanar: Array
    stacked: Array
    valid: Array
    center_distance: Array
    normal_cosine: Array


def geometric_contacts(
    positions, descriptor: NucleotideGDescriptor, criteria: GeometricContactCriteria
) -> GeometricContactEvaluation:
    frames = base_frames(
        positions, descriptor.binding, image_policy=descriptor.image_policy
    )
    i, j = descriptor.pair_indices[:, 0], descriptor.pair_indices[:, 1]
    relative = (frames.centers[j] - frames.centers[i]) * descriptor.length_to_angstrom
    local_i = contract("pi,pij->pj", relative, frames.axes[i])
    local_j = contract("pi,pij->pj", -relative, frames.axes[j])
    distance = jnp.sqrt(jnp.sum(relative**2, axis=-1))
    cosine = jnp.sum(frames.axes[i, :, 2] * frames.axes[j, :, 2], axis=-1)
    valid = frames.valid[i] & frames.valid[j]
    common = (
        valid
        & (distance <= criteria.maximum_distance)
        & (jnp.abs(cosine) >= criteria.minimum_abs_normal_cosine)
    )
    heights = jnp.stack((jnp.abs(local_i[:, 2]), jnp.abs(local_j[:, 2])), axis=-1)
    coplanar = common & jnp.all(heights <= criteria.coplanar_height, axis=-1)
    stacked = (
        common
        & jnp.all(
            (heights >= criteria.stacking_height[0])
            & (heights <= criteria.stacking_height[1]),
            axis=-1,
        )
        & (jnp.sum(local_i[:, :2] ** 2, axis=-1) <= criteria.stacking_lateral_radius**2)
        & (jnp.sum(local_j[:, :2] ** 2, axis=-1) <= criteria.stacking_lateral_radius**2)
    )
    return GeometricContactEvaluation(coplanar, stacked, valid, distance, cosine)


def contact_interaction_graph(
    result, descriptor, criteria, *, source_id: str
) -> BaseInteractionGraph:
    """Host annotation retaining directed contact support, not canonical pairing."""
    keys = descriptor.binding.construct.nucleotide_keys
    records = []
    for (i, j), coplanar, stacked in zip(
        np.asarray(descriptor.pair_indices),
        np.asarray(result.coplanar),
        np.asarray(result.stacked),
        strict=True,
    ):
        if coplanar or stacked:
            records.append(
                BaseInteraction(
                    keys[i],
                    keys[j],
                    "geometric-contact",
                    criteria.name + (":stacked" if stacked else ":coplanar"),
                    source_id,
                )
            )
    return BaseInteractionGraph(descriptor.binding.construct, tuple(records))


__all__ = [
    "GeometricContactCriteria",
    "GeometricContactEvaluation",
    "geometric_contacts",
    "contact_interaction_graph",
]
