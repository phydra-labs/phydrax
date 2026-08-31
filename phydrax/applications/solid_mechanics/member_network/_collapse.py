#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._strict import StrictModule
from ...._trainable import NonTrainableState


class ImperfectionSource(IntEnum):
    EXPLICIT = 0
    EIGENMODE = 1
    MEMBER_BOW = 2
    SECTION_MODE = 3
    FABRICATION_SAMPLE = 4


class StructuralImperfection(StrictModule, NonTrainableState):
    displacement: Array
    source: ImperfectionSource = eqx.field(static=True)
    amplitude: float = eqx.field(static=True)
    source_id: str = eqx.field(static=True)

    def __init__(
        self,
        shape: ArrayLike,
        source: ImperfectionSource,
        amplitude: float,
        /,
        *,
        source_id: str = "structural-imperfection",
    ):
        shape_ = jnp.asarray(shape)
        norm = jnp.sqrt(jnp.sum(shape_ * shape_))
        if bool(~jnp.isfinite(norm) | (norm <= 0.0)):
            raise ValueError("Imperfection shape must be finite and nonzero.")
        self.displacement = float(amplitude) * shape_ / norm
        self.source = ImperfectionSource(source)
        self.amplitude = float(amplitude)
        self.source_id = str(source_id)

    def apply(self, coordinates: ArrayLike, /) -> Array:
        coordinates_ = jnp.asarray(coordinates)
        if coordinates_.shape != self.displacement.shape:
            raise ValueError("Imperfection and coordinate shapes do not match.")
        return coordinates_ + self.displacement


class CollapseEventType(IntEnum):
    NONE = 0
    LIMIT_POINT = 1
    BRANCH_POINT = 2
    TANGENT_INSTABILITY = 3
    PLASTIC_MECHANISM = 4
    STRAIN_LIMIT = 5
    ROTATION_LIMIT = 6
    FRACTURE = 7
    CONTACT_TRANSITION = 8
    UNBOUNDED_RESPONSE = 9


class CollapseEvidence(StrictModule):
    event: Array
    load_parameter: Array
    governing_index: Array
    minimum_tangent_eigenvalue: Array
    yielded_fraction: Array
    fractured_fraction: Array
    state_norm: Array
    physically_classified: Array

    @property
    def collapsed(self) -> Array:
        return self.event != int(CollapseEventType.NONE)


def classify_collapse(
    load_parameter: ArrayLike,
    tangent_eigenvalues: ArrayLike,
    /,
    *,
    yielded: ArrayLike | None = None,
    fractured: ArrayLike | None = None,
    strain_utilization: ArrayLike | None = None,
    rotation_utilization: ArrayLike | None = None,
    fold_detected: ArrayLike = False,
    branch_detected: ArrayLike = False,
    state_norm: ArrayLike = 0.0,
    unbounded_threshold: float = jnp.inf,
) -> CollapseEvidence:
    eigenvalues = jnp.asarray(tangent_eigenvalues)
    minimum = jnp.min(eigenvalues)
    yielded_ = (
        jnp.zeros((1,), dtype=bool)
        if yielded is None
        else jnp.asarray(yielded, dtype=bool)
    )
    fractured_ = (
        jnp.zeros((1,), dtype=bool)
        if fractured is None
        else jnp.asarray(fractured, dtype=bool)
    )
    strain = (
        jnp.zeros((1,)) if strain_utilization is None else jnp.asarray(strain_utilization)
    )
    rotation = (
        jnp.zeros((1,))
        if rotation_utilization is None
        else jnp.asarray(rotation_utilization)
    )
    norm = jnp.asarray(state_norm)
    event = jnp.where(
        jnp.any(fractured_),
        int(CollapseEventType.FRACTURE),
        jnp.where(
            jnp.any(strain >= 1.0),
            int(CollapseEventType.STRAIN_LIMIT),
            jnp.where(
                jnp.any(rotation >= 1.0),
                int(CollapseEventType.ROTATION_LIMIT),
                jnp.where(
                    norm >= unbounded_threshold,
                    int(CollapseEventType.UNBOUNDED_RESPONSE),
                    jnp.where(
                        branch_detected,
                        int(CollapseEventType.BRANCH_POINT),
                        jnp.where(
                            fold_detected,
                            int(CollapseEventType.LIMIT_POINT),
                            jnp.where(
                                minimum <= 0.0,
                                int(CollapseEventType.TANGENT_INSTABILITY),
                                jnp.where(
                                    jnp.all(yielded_),
                                    int(CollapseEventType.PLASTIC_MECHANISM),
                                    int(CollapseEventType.NONE),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)
    candidates = jnp.concatenate((strain.reshape((-1,)), rotation.reshape((-1,))))
    governing = jnp.argmax(candidates).astype(jnp.int32)
    return CollapseEvidence(
        event,
        jnp.asarray(load_parameter),
        governing,
        minimum,
        jnp.mean(yielded_.astype(jnp.float64)),
        jnp.mean(fractured_.astype(jnp.float64)),
        norm,
        event != int(CollapseEventType.UNBOUNDED_RESPONSE),
    )


__all__ = [
    "CollapseEventType",
    "CollapseEvidence",
    "ImperfectionSource",
    "StructuralImperfection",
    "classify_collapse",
]
