#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ._reference import rotation_vector_matrix


class SectionOrientationSource(IntEnum):
    EXPLICIT_DIRECTOR = 0
    CAD_FRAME = 1
    SURFACE_NORMAL = 2
    JOINT_FRAME = 3
    BISHOP_TRANSPORT = 4
    OPTIMIZED = 5


class OrientationHolonomyEvidence(StrictModule):
    loop_rotation: Array
    mismatch_angle: Array
    compatible: Array
    tolerance: float = eqx.field(static=True)


class SectionOrientationField(StrictModule, NonTrainableState):
    """Member frames, sources, continuity groups, and singularity evidence."""

    frames: Array
    sources: Array
    continuity_group: Array
    director_margin: Array
    orientation_id: str = eqx.field(static=True)

    def __init__(
        self,
        frames: ArrayLike,
        sources: ArrayLike,
        /,
        *,
        continuity_group: ArrayLike | None = None,
        orientation_id: str = "section-orientation-field",
    ):
        frames_ = jnp.asarray(frames)
        sources_ = jnp.asarray(sources, dtype=jnp.int32)
        if frames_.ndim != 3 or frames_.shape[-2:] != (3, 3):
            raise ValueError("Section frames must have shape (members, 3, 3).")
        if sources_.shape != (frames_.shape[0],):
            raise ValueError("Orientation sources must match the member axis.")
        orthogonality = frames_.swapaxes(-1, -2) @ frames_
        error = jnp.max(
            jnp.abs(orthogonality - jnp.eye(3, dtype=frames_.dtype)), axis=(-2, -1)
        )
        determinant = jnp.linalg.det(frames_)
        if bool(jnp.any(error > 1.0e-8) | jnp.any(determinant <= 0.0)):
            raise ValueError("Section frames must be proper orthogonal matrices.")
        groups = (
            jnp.arange(frames_.shape[0], dtype=jnp.int32)
            if continuity_group is None
            else jnp.asarray(continuity_group, dtype=jnp.int32)
        )
        if groups.shape != sources_.shape:
            raise ValueError("continuity_group must match the member axis.")
        self.frames = frames_
        self.sources = sources_
        self.continuity_group = groups
        self.director_margin = 1.0 - error
        self.orientation_id = str(orientation_id)


def explicit_section_orientations(
    member_vectors: ArrayLike,
    directors: ArrayLike,
    /,
    *,
    source: SectionOrientationSource = SectionOrientationSource.EXPLICIT_DIRECTOR,
) -> SectionOrientationField:
    vectors = jnp.asarray(member_vectors)
    directors_ = jnp.asarray(directors, dtype=vectors.dtype)
    if vectors.shape != directors_.shape or vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError("Member vectors and directors must have shape (members, 3).")
    first_norm = jnp.sqrt(jnp.sum(vectors * vectors, axis=-1))
    first = vectors / first_norm[:, None]
    projected = directors_ - jnp.sum(directors_ * first, axis=-1)[:, None] * first
    margin = jnp.sqrt(jnp.sum(projected * projected, axis=-1))
    if bool(jnp.any(first_norm <= 0.0) | jnp.any(margin <= 1.0e-10)):
        raise ValueError("Members and projected directors must be nondegenerate.")
    second = projected / margin[:, None]
    third = jnp.cross(first, second)
    frames = jnp.stack((first, second, third), axis=-1)
    return SectionOrientationField(
        frames,
        jnp.full((vectors.shape[0],), int(source), dtype=jnp.int32),
    )


def parallel_transport_orientations(
    points: ArrayLike,
    initial_director: ArrayLike,
    /,
    *,
    closed: bool = False,
    tolerance: float = 1.0e-8,
) -> tuple[SectionOrientationField, OrientationHolonomyEvidence | None]:
    xyz = jnp.asarray(points)
    director = jnp.asarray(initial_director, dtype=xyz.dtype)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or director.shape != (3,):
        raise ValueError("Chain points and initial director have invalid shapes.")
    edges = xyz[1:] - xyz[:-1]
    tangents = edges / jnp.sqrt(jnp.sum(edges * edges, axis=-1))[:, None]
    director = director - jnp.dot(director, tangents[0]) * tangents[0]
    director = director / jnp.sqrt(jnp.dot(director, director))
    directors = [director]
    frames = []
    for index, tangent in enumerate(np.asarray(tangents)):
        tangent_ = jnp.asarray(tangent, dtype=xyz.dtype)
        if index:
            previous = tangents[index - 1]
            axis = jnp.cross(previous, tangent_)
            sine = jnp.sqrt(jnp.dot(axis, axis))
            cosine = jnp.dot(previous, tangent_)
            if float(cosine) <= -1.0 + tolerance:
                raise ValueError(
                    "Parallel transport is singular at antiparallel tangents."
                )
            if float(sine) > tolerance:
                axis = axis / sine
                angle = jnp.arctan2(sine, cosine)
                rotation = rotation_vector_matrix((axis * angle)[None, :])[0]
                director = rotation @ director
            director = director - jnp.dot(director, tangent_) * tangent_
            director = director / jnp.sqrt(jnp.dot(director, director))
            directors.append(director)
        third = jnp.cross(tangent_, director)
        frames.append(jnp.stack((tangent_, director, third), axis=-1))
    field = SectionOrientationField(
        jnp.stack(tuple(frames)),
        jnp.full((len(frames),), int(SectionOrientationSource.BISHOP_TRANSPORT)),
        continuity_group=jnp.zeros((len(frames),), dtype=jnp.int32),
        orientation_id="bishop-parallel-transport",
    )
    evidence = None
    if closed:
        first = field.frames[0]
        last = field.frames[-1]
        relative = first.T @ last
        trace = jnp.clip((jnp.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
        angle = jnp.arccos(trace)
        evidence = OrientationHolonomyEvidence(
            relative,
            angle,
            angle <= tolerance,
            float(tolerance),
        )
    return field, evidence


__all__ = [
    "OrientationHolonomyEvidence",
    "SectionOrientationField",
    "SectionOrientationSource",
    "explicit_section_orientations",
    "parallel_transport_orientations",
]
