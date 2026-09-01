#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Sequence

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._strict import StrictModule
from ...topology import PackedPersistenceDiagram
from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)


class MorphologyStatus(IntEnum):
    OK = 0
    CAPACITY_EXCEEDED = 1


@dataclass(frozen=True, slots=True)
class MorphologyPlan:
    max_objects: int
    pixel_size: tuple[float, float] = (1.0, 1.0)
    origin: tuple[float, float] = (0.0, 0.0)

    def __init__(
        self,
        max_objects: int,
        /,
        *,
        pixel_size: Sequence[float] = (1.0, 1.0),
        origin: Sequence[float] = (0.0, 0.0),
    ):
        maximum = int(max_objects)
        spacing = tuple(float(value) for value in pixel_size)
        origin_ = tuple(float(value) for value in origin)
        if maximum <= 0:
            raise ValueError("max_objects must be positive.")
        if len(spacing) != 2 or any(
            not np.isfinite(value) or value <= 0.0 for value in spacing
        ):
            raise ValueError("pixel_size must contain two finite positive values.")
        if len(origin_) != 2 or any(not np.isfinite(value) for value in origin_):
            raise ValueError("origin must contain two finite values.")
        object.__setattr__(self, "max_objects", maximum)
        object.__setattr__(self, "pixel_size", spacing)
        object.__setattr__(self, "origin", origin_)


class MorphologyEvidence(StrictModule):
    required_objects: Array
    configured_objects: Array
    foreground_pixels: Array


_MORPHOLOGY_CONTRACT = BioinformaticsMethodContract(
    "bounded_label_morphology_and_cubical_euler_summary",
    MethodKind.EXACT_MODEL,
    ExecutionKind.FLOATING_POINT_DIRECT,
    DifferentiationKind.NONE,
    OutputKind.STRUCTURED,
    conditioning_statement=(
        "Labels denote 2D closed pixel cells with four-neighbor cubical adjacency and "
        "the declared physical pixel size."
    ),
    truncation_statement=(
        "No object summaries are valid when the number of positive labels exceeds capacity."
    ),
    capacity_semantics=(
        "Every per-object output has MorphologyPlan.max_objects slots and an explicit mask."
    ),
    assumptions=("Zero is background and positive integers identify objects.",),
    nondifferentiable_outputs=(
        "labels",
        "object_valid",
        "euler_characteristic",
        "status",
    ),
)


class MorphologySummary(StrictModule):
    labels: Array
    object_valid: Array
    area: Array
    perimeter: Array
    centroid: Array
    covariance: Array
    eccentricity: Array
    compactness: Array
    euler_characteristic: Array
    valid: Array
    status: Array
    evidence: MorphologyEvidence
    method_contract: BioinformaticsMethodContract


def _object_labels(labels: Array, capacity: int, /) -> tuple[Array, Array]:
    ordered = jnp.sort(labels.reshape((-1,)))
    unique = (ordered > 0) & jnp.concatenate(
        (jnp.ones((1,), dtype=bool), ordered[1:] != ordered[:-1])
    )
    required = jnp.sum(unique, dtype=jnp.int32)
    positions = jnp.nonzero(unique, size=capacity, fill_value=0)[0]
    selected = ordered[positions]
    valid = jnp.arange(capacity) < required
    return jnp.where(valid, selected, 0), required


def summarize_label_morphology(
    label_image: Any,
    plan: MorphologyPlan,
    /,
) -> MorphologySummary:
    """Return fixed-capacity metric and cubical Euler summaries for a 2D label image."""
    if not isinstance(plan, MorphologyPlan):
        raise TypeError("plan must be a MorphologyPlan.")
    labels = jnp.asarray(label_image, dtype=jnp.int32)
    if labels.ndim != 2 or int(labels.shape[0]) < 1 or int(labels.shape[1]) < 1:
        raise ValueError("label_image must be a non-empty rank-2 array.")
    if np.any(np.asarray(labels) < 0):
        raise ValueError("Morphology labels must be non-negative.")
    selected, required = _object_labels(labels, plan.max_objects)
    overflow = required > plan.max_objects
    object_valid = (jnp.arange(plan.max_objects) < required) & ~overflow
    masks = labels[None, :, :] == selected[:, None, None]
    masks = masks & object_valid[:, None, None]
    pixel_count = jnp.sum(masks, axis=(1, 2))
    spacing_y, spacing_x = plan.pixel_size
    pixel_area = spacing_y * spacing_x
    area = pixel_count * pixel_area

    vertical_edges = (
        jnp.sum(masks[:, :, 0], axis=1)
        + jnp.sum(masks[:, :, -1], axis=1)
        + jnp.sum(masks[:, :, 1:] != masks[:, :, :-1], axis=(1, 2))
    )
    horizontal_edges = (
        jnp.sum(masks[:, 0, :], axis=1)
        + jnp.sum(masks[:, -1, :], axis=1)
        + jnp.sum(masks[:, 1:, :] != masks[:, :-1, :], axis=(1, 2))
    )
    perimeter = vertical_edges * spacing_y + horizontal_edges * spacing_x

    y = plan.origin[0] + (jnp.arange(labels.shape[0], dtype=float) + 0.5) * spacing_y
    x = plan.origin[1] + (jnp.arange(labels.shape[1], dtype=float) + 0.5) * spacing_x
    denominator = jnp.maximum(pixel_count, 1)[:, None]
    centroid_y = jnp.sum(masks * y[None, :, None], axis=(1, 2))
    centroid_x = jnp.sum(masks * x[None, None, :], axis=(1, 2))
    centroid = jnp.stack((centroid_y, centroid_x), axis=1) / denominator
    dy = y[None, :, None] - centroid[:, 0, None, None]
    dx = x[None, None, :] - centroid[:, 1, None, None]
    cov_yy = jnp.sum(masks * dy * dy, axis=(1, 2)) / jnp.maximum(pixel_count, 1)
    cov_xx = jnp.sum(masks * dx * dx, axis=(1, 2)) / jnp.maximum(pixel_count, 1)
    cov_yx = jnp.sum(masks * dy * dx, axis=(1, 2)) / jnp.maximum(pixel_count, 1)
    covariance = jnp.stack(
        (
            jnp.stack((cov_yy, cov_yx), axis=1),
            jnp.stack((cov_yx, cov_xx), axis=1),
        ),
        axis=1,
    )
    discriminant = jnp.sqrt(jnp.maximum((cov_yy - cov_xx) ** 2 + 4.0 * cov_yx**2, 0.0))
    eigen_max = 0.5 * (cov_yy + cov_xx + discriminant)
    eigen_min = 0.5 * (cov_yy + cov_xx - discriminant)
    eccentricity = jnp.sqrt(
        jnp.maximum(1.0 - eigen_min / jnp.maximum(eigen_max, 1.0e-30), 0.0)
    )
    eccentricity = jnp.where(eigen_max > 0.0, eccentricity, 0.0)
    compactness = jnp.where(perimeter > 0.0, 4.0 * jnp.pi * area / perimeter**2, 0.0)

    shared_edges = jnp.sum(masks[:, 1:, :] & masks[:, :-1, :], axis=(1, 2)) + jnp.sum(
        masks[:, :, 1:] & masks[:, :, :-1], axis=(1, 2)
    )
    padded = jnp.pad(masks, ((0, 0), (1, 1), (1, 1)))
    occupied_vertices = (
        padded[:, :-1, :-1] | padded[:, 1:, :-1] | padded[:, :-1, 1:] | padded[:, 1:, 1:]
    )
    vertex_count = jnp.sum(occupied_vertices, axis=(1, 2))
    edge_count = 4 * pixel_count - shared_edges
    euler = (vertex_count - edge_count + pixel_count).astype(jnp.int32)

    evidence = MorphologyEvidence(
        required_objects=required,
        configured_objects=jnp.asarray(plan.max_objects, dtype=jnp.int32),
        foreground_pixels=jnp.sum(labels > 0, dtype=jnp.int32),
    )
    return MorphologySummary(
        labels=jnp.where(object_valid, selected, 0),
        object_valid=object_valid,
        area=jnp.where(object_valid, area, 0.0),
        perimeter=jnp.where(object_valid, perimeter, 0.0),
        centroid=jnp.where(object_valid[:, None], centroid, 0.0),
        covariance=jnp.where(object_valid[:, None, None], covariance, 0.0),
        eccentricity=jnp.where(object_valid, eccentricity, 0.0),
        compactness=jnp.where(object_valid, compactness, 0.0),
        euler_characteristic=jnp.where(object_valid, euler, 0),
        valid=~overflow,
        status=jnp.where(
            overflow,
            int(MorphologyStatus.CAPACITY_EXCEEDED),
            int(MorphologyStatus.OK),
        ).astype(jnp.int32),
        evidence=evidence,
        method_contract=_MORPHOLOGY_CONTRACT,
    )


class PersistenceTopologyEvidence(StrictModule):
    interval_count: Array
    essential_interval_count: Array
    threshold: Array


_TOPOLOGY_CONTRACT = BioinformaticsMethodContract(
    "native_persistence_topology_summary",
    MethodKind.EXACT_MODEL,
    ExecutionKind.EXACT_DISCRETE,
    DifferentiationKind.NONE,
    OutputKind.STRUCTURED,
    conditioning_statement=(
        "Betti counts are evaluated from one native packed persistence diagram at the "
        "declared filtration threshold."
    ),
    truncation_statement=(
        "All active intervals in the preflighted native packed diagram are included."
    ),
    capacity_semantics=(
        "The upstream PackedPersistenceDiagram constructor rejects interval overflow."
    ),
    nondifferentiable_outputs=("betti_numbers", "essential_counts", "status", "evidence"),
)


class PersistenceTopologySummary(StrictModule):
    betti_numbers: Array
    total_finite_persistence: Array
    essential_counts: Array
    valid: Array
    status: Array
    evidence: PersistenceTopologyEvidence
    method_contract: BioinformaticsMethodContract


def summarize_persistence_topology(
    diagram: PackedPersistenceDiagram,
    threshold: Any,
    /,
    *,
    max_degree: int,
) -> PersistenceTopologySummary:
    """Summarize exact native persistence intervals without reconstructing topology."""
    if not isinstance(diagram, PackedPersistenceDiagram):
        raise TypeError("diagram must be a PackedPersistenceDiagram.")
    degree_count = int(max_degree) + 1
    if degree_count <= 0:
        raise ValueError("max_degree must be non-negative.")
    threshold_ = jnp.asarray(threshold, dtype=diagram.birth_values.dtype)
    if threshold_.shape != () or not np.isfinite(float(threshold_)):
        raise ValueError("threshold must be a finite scalar.")
    active_at_threshold = (
        diagram.active_mask
        & (diagram.birth_values <= threshold_)
        & (~diagram.has_finite_death | (diagram.death_values > threshold_))
    )
    degrees = jnp.arange(degree_count, dtype=jnp.int32)[:, None]
    degree_match = diagram.degrees[None, :] == degrees
    betti = jnp.sum(degree_match & active_at_threshold[None, :], axis=1, dtype=jnp.int32)
    finite = diagram.active_mask & diagram.has_finite_death
    persistence = jnp.where(finite, diagram.death_values - diagram.birth_values, 0.0)
    total = jnp.sum(jnp.where(degree_match, persistence[None, :], 0.0), axis=1)
    essential = jnp.sum(
        degree_match & diagram.active_mask[None, :] & ~diagram.has_finite_death[None, :],
        axis=1,
        dtype=jnp.int32,
    )
    evidence = PersistenceTopologyEvidence(
        interval_count=diagram.interval_count,
        essential_interval_count=jnp.sum(
            diagram.active_mask & ~diagram.has_finite_death, dtype=jnp.int32
        ),
        threshold=threshold_,
    )
    return PersistenceTopologySummary(
        betti_numbers=betti,
        total_finite_persistence=total,
        essential_counts=essential,
        valid=jnp.asarray(True),
        status=jnp.asarray(0, dtype=jnp.int32),
        evidence=evidence,
        method_contract=_TOPOLOGY_CONTRACT,
    )


__all__ = [
    "MorphologyEvidence",
    "MorphologyPlan",
    "MorphologyStatus",
    "MorphologySummary",
    "PersistenceTopologyEvidence",
    "PersistenceTopologySummary",
    "summarize_label_morphology",
    "summarize_persistence_topology",
]
