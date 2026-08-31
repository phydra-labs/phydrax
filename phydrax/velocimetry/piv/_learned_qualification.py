#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..imaging._types import DenseDisplacementField2D
from ..imaging._warp import image_coordinates
from ._learned_model import AbstractDensePIVModel
from ._learned_primitives import MultiScaleRobustPIVLoss, PIVLossResult
from ._learned_training import evaluate_learned_piv, LearnedPIVDataset


class LearnedPIVQualificationResult(StrictModule, NonTrainableState):
    """Held-out dense fields plus reference-backed aggregate evidence."""

    fields: tuple[DenseDisplacementField2D, ...]
    loss: PIVLossResult
    endpoint_error: Array
    reference_valid_count: Array
    predicted_valid_fraction: Array
    scenario_ids: tuple[str, ...] = eqx.field(static=True)
    qualification_id: str = eqx.field(static=True)


def qualify_learned_piv(
    model: AbstractDensePIVModel,
    held_out: LearnedPIVDataset,
    /,
    *,
    loss: MultiScaleRobustPIVLoss | None = None,
) -> LearnedPIVQualificationResult:
    """Predict and score named held-out scenarios on their canonical image grid.

    The adapter only transfers displacement and support into
    ``DenseDisplacementField2D``. It does not invent confidence, variance, or any
    uncertainty-bearing payload.
    """

    if not isinstance(model, AbstractDensePIVModel):
        raise TypeError("model must satisfy AbstractDensePIVModel.")
    if not isinstance(held_out, LearnedPIVDataset):
        raise TypeError("held_out must be a LearnedPIVDataset.")
    if held_out.partition != "held-out":
        raise ValueError("Qualification requires a dataset explicitly marked 'held-out'.")
    if held_out.geometry is None:
        raise ValueError("Held-out qualification requires an ImageGeometry2D.")
    if held_out.target_forward_rc is None:
        raise ValueError(
            "Held-out qualification requires reference forward displacement."
        )
    loss_ = MultiScaleRobustPIVLoss() if loss is None else loss
    if not isinstance(loss_, MultiScaleRobustPIVLoss):
        raise TypeError("loss must be a MultiScaleRobustPIVLoss or None.")
    if held_out.image_shape != model.plan.image_shape:
        raise ValueError("Held-out image shape does not match the model plan.")
    if held_out.channel_count != model.plan.input_channels:
        raise ValueError("Held-out channels do not match the model plan.")

    positions = image_coordinates(held_out.geometry)
    fields: list[DenseDisplacementField2D] = []
    endpoint_errors: list[Array] = []
    reference_counts: list[Array] = []
    valid_fractions: list[Array] = []
    for index, scenario_id in enumerate(held_out.scenario_ids):
        prediction = model(
            model.plan.prepare(
                held_out.first_images[index],
                held_out.second_images[index],
                first_valid=held_out.first_valid[index],
                second_valid=held_out.second_valid[index],
            )
        )
        field = DenseDisplacementField2D(
            positions,
            prediction.displacement_rc,
            prediction.valid,
            geometry_id=held_out.geometry.geometry_id,
            provenance=(
                f"held-out-dataset:{held_out.dataset_id}",
                f"scenario:{scenario_id}",
                f"learned-piv-architecture:{model.architecture_id}",
            ),
        )
        fields.append(field)
        reference_valid = (
            held_out.first_valid[index]
            & prediction.valid
            & (
                jnp.ones_like(prediction.valid)
                if held_out.target_valid is None
                else held_out.target_valid[index]
            )
        )
        endpoint = jnp.sqrt(
            jnp.sum(
                jnp.square(
                    prediction.displacement_rc - held_out.target_forward_rc[index]
                ),
                axis=-1,
            )
        )
        count = jnp.sum(reference_valid)
        endpoint_sum = jnp.sum(jnp.where(reference_valid, endpoint, 0.0))
        endpoint_errors.append(
            jnp.where(
                count > 0,
                endpoint_sum / jnp.maximum(count, 1),
                0.0,
            )
        )
        reference_counts.append(count)
        valid_fractions.append(jnp.mean(prediction.valid.astype(float)))

    aggregate_loss = evaluate_learned_piv(model, held_out, loss_)
    endpoint_error = jnp.stack(endpoint_errors)
    reference_valid_count = jnp.stack(reference_counts)
    predicted_valid_fraction = jnp.stack(valid_fractions)
    parameter_sha256 = array_tree_fingerprint(model)["sha256"]
    qualification_id = canonical_fingerprint(
        {
            "kind": "learned-piv-held-out-qualification",
            "architecture_id": model.architecture_id,
            "parameter_sha256": parameter_sha256,
            "dataset_id": held_out.dataset_id,
            "loss_id": loss_.loss_id,
            "scenario_ids": held_out.scenario_ids,
        }
    )
    return LearnedPIVQualificationResult(
        fields=tuple(fields),
        loss=aggregate_loss,
        endpoint_error=endpoint_error,
        reference_valid_count=reference_valid_count,
        predicted_valid_fraction=predicted_valid_fraction,
        scenario_ids=held_out.scenario_ids,
        qualification_id=qualification_id,
    )


__all__ = ["LearnedPIVQualificationResult", "qualify_learned_piv"]
