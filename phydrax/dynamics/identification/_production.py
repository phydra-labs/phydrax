#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._system import ContinuousSystem, DiscreteSystem
from .._trajectory import StateLayout


TransformKind: TypeAlias = Literal["affine", "log-affine"]


class IdentificationStateTransform(StrictModule, NonTrainableState):
    """Invertible train-only physical-to-identification state transform."""

    offset: Array
    scale: Array
    physical_layout: StateLayout
    transformed_layout: StateLayout
    kind: TransformKind = eqx.field(static=True)
    unit_contract_id: str = eqx.field(static=True)
    partition_id: str = eqx.field(static=True)
    source_artifact_ids: tuple[str, ...] = eqx.field(static=True)
    transform_id: str = eqx.field(static=True)

    def __init__(
        self,
        physical_layout: StateLayout,
        offset: ArrayLike,
        scale: ArrayLike,
        /,
        *,
        kind: TransformKind = "affine",
        unit_contract_id: str,
        partition_id: str,
        source_artifact_ids: Sequence[str],
    ):
        if not isinstance(physical_layout, StateLayout):
            raise TypeError("physical_layout must be a StateLayout.")
        if kind not in ("affine", "log-affine"):
            raise ValueError("kind must be 'affine' or 'log-affine'.")
        offset_ = jnp.asarray(offset)
        scale_ = jnp.asarray(scale, dtype=offset_.dtype)
        if (
            offset_.shape != physical_layout.shape
            or scale_.shape != physical_layout.shape
        ):
            raise ValueError(
                "Transform offset and scale must match the physical state layout."
            )
        host_scale = np.asarray(scale_)
        if (
            np.any(~np.isfinite(np.asarray(offset_)))
            or np.any(~np.isfinite(host_scale))
            or np.any(host_scale <= 0.0)
        ):
            raise ValueError(
                "Transform offset and scale must be finite with positive scale."
            )
        units = str(unit_contract_id)
        partition = str(partition_id)
        sources = tuple(str(value) for value in source_artifact_ids)
        if (
            not units
            or not partition
            or not sources
            or any(not value for value in sources)
        ):
            raise ValueError("Transform identities must be non-empty.")
        transformed = StateLayout(
            physical_layout.shape,
            axes=physical_layout.axes,
            component_names=physical_layout.component_names,
            layout_id=canonical_fingerprint(
                {
                    "kind": "transformed-state-layout",
                    "source": physical_layout.layout_id,
                    "transform_kind": kind,
                }
            ),
        )
        self.offset = offset_
        self.scale = scale_
        self.physical_layout = physical_layout
        self.transformed_layout = transformed
        self.kind = kind
        self.unit_contract_id = units
        self.partition_id = partition
        self.source_artifact_ids = sources
        self.transform_id = canonical_fingerprint(
            {
                "kind": "identification-state-transform",
                "transform_kind": kind,
                "physical_layout": physical_layout.layout_id,
                "transformed_layout": transformed.layout_id,
                "units": units,
                "partition": partition,
                "sources": list(sources),
                "content": array_tree_fingerprint({"offset": offset_, "scale": scale_})[
                    "sha256"
                ],
            }
        )

    def forward(self, physical_state: ArrayLike, /) -> Array:
        value = jnp.asarray(physical_state)
        shape = self.physical_layout.shape
        if shape and value.shape[-len(shape) :] != shape:
            raise ValueError("Physical state trailing shape does not match its layout.")
        shifted = value - self.offset
        if self.kind == "log-affine":
            shifted = eqx.error_if(
                shifted,
                jnp.any(shifted <= 0.0),
                "log-affine transform requires state > offset",
            )
            shifted = jnp.log(shifted)
        return shifted / self.scale

    def inverse(self, transformed_state: ArrayLike, /) -> Array:
        value = jnp.asarray(transformed_state)
        shifted = self.scale * value
        if self.kind == "log-affine":
            shifted = jnp.exp(shifted)
        return shifted + self.offset

    def tangent_pushforward(
        self, physical_state: ArrayLike, tangent: ArrayLike, /
    ) -> Array:
        state = jnp.asarray(physical_state)
        direction = jnp.asarray(tangent)
        if state.shape != direction.shape:
            raise ValueError("State and tangent must share shape.")
        denominator = (
            self.scale if self.kind == "affine" else self.scale * (state - self.offset)
        )
        return direction / denominator


class IdentifiedDynamicsArtifact(StrictModule, NonTrainableState):
    transform: IdentificationStateTransform
    system: ContinuousSystem | DiscreteSystem
    partition_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    formulation_id: str = eqx.field(static=True)
    evidence_ids: tuple[str, ...] = eqx.field(static=True)
    artifact_id: str = eqx.field(static=True)

    def __init__(
        self,
        transform: IdentificationStateTransform,
        system: ContinuousSystem | DiscreteSystem,
        /,
        *,
        support_id: str,
        formulation_id: str,
        evidence_ids: Sequence[str],
    ):
        if not isinstance(transform, IdentificationStateTransform):
            raise TypeError("transform must be IdentificationStateTransform.")
        if not isinstance(system, (ContinuousSystem, DiscreteSystem)):
            raise TypeError("system must be ContinuousSystem or DiscreteSystem.")
        support = str(support_id)
        formulation = str(formulation_id)
        evidence = tuple(str(value) for value in evidence_ids)
        if (
            not support
            or not formulation
            or not evidence
            or any(not value for value in evidence)
        ):
            raise ValueError("Identified dynamics identities must be non-empty.")
        if system.state_layout.layout_id != transform.transformed_layout.layout_id:
            raise ValueError(
                "Identified system must use the exact transformed state layout."
            )
        self.transform = transform
        self.system = system
        self.partition_id = transform.partition_id
        self.support_id = support
        self.formulation_id = formulation
        self.evidence_ids = evidence
        self.artifact_id = canonical_fingerprint(
            {
                "kind": "identified-dynamics-artifact",
                "transform": transform.transform_id,
                "system": system.system_id,
                "partition": transform.partition_id,
                "support": support,
                "formulation": formulation,
                "evidence": list(evidence),
            }
        )


class IdentifiedDynamicsSelection(StrictModule, NonTrainableState):
    selected_index: int = eqx.field(static=True)
    validation_scores: Array
    candidate_ids: tuple[str, ...] = eqx.field(static=True)
    selection_id: str = eqx.field(static=True)


def select_identified_dynamics(
    candidates: Sequence[IdentifiedDynamicsArtifact],
    validation_scores: ArrayLike,
    /,
) -> tuple[IdentifiedDynamicsArtifact, IdentifiedDynamicsSelection]:
    models = tuple(candidates)
    scores = jnp.asarray(validation_scores)
    if not models or scores.shape != (len(models),):
        raise ValueError(
            "Candidates and validation scores must be non-empty and aligned."
        )
    if any(not isinstance(model, IdentifiedDynamicsArtifact) for model in models):
        raise TypeError("Every candidate must be an IdentifiedDynamicsArtifact.")
    host = np.asarray(scores)
    if np.any(~np.isfinite(host)):
        raise ValueError("Validation scores must be finite.")
    reference = models[0]
    reference_input = (
        None
        if reference.system.input_layout is None
        else reference.system.input_layout.layout_id
    )
    for model in models[1:]:
        model_input = (
            None
            if model.system.input_layout is None
            else model.system.input_layout.layout_id
        )
        compatible = (
            model.partition_id == reference.partition_id
            and model.transform.transform_id == reference.transform.transform_id
            and type(model.system) is type(reference.system)
            and model.system.state_layout.layout_id
            == reference.system.state_layout.layout_id
            and model_input == reference_input
            and model.formulation_id == reference.formulation_id
            and model.support_id == reference.support_id
        )
        if not compatible:
            raise ValueError(
                "Identification candidates must share transform, system, input, "
                "formulation, support, and partition contracts."
            )
    index = int(np.argmin(host))
    ids = tuple(model.artifact_id for model in models)
    selection = IdentifiedDynamicsSelection(
        index,
        scores,
        ids,
        canonical_fingerprint(
            {
                "kind": "identified-dynamics-selection",
                "candidates": list(ids),
                "scores": host.tolist(),
                "selected": index,
            }
        ),
    )
    return models[index], selection


__all__ = [
    "IdentificationStateTransform",
    "IdentifiedDynamicsArtifact",
    "IdentifiedDynamicsSelection",
    "TransformKind",
    "select_identified_dynamics",
]
