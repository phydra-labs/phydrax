#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from jaxtyping import Array, ArrayLike, Key

from ..._doc import DOC_KEY0
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import combine_trainable, NonTrainableState, partition_trainable
from ..._training import TrainingCallback, TrainingController, TrainingProgress
from ..imaging._types import ImageGeometry2D
from ._learned_model import AbstractDensePIVModel
from ._learned_primitives import (
    MultiScaleRobustPIVLoss,
    PIV_LOSS_INSUFFICIENT_SUPPORT,
    PIV_LOSS_SUCCESS,
    PIVLossResult,
)


DatasetPartition = Literal["training", "validation", "held-out"]


class LearnedPIVDataset(StrictModule, NonTrainableState):
    """Fixed-capacity NHWC learned-PIV cases with explicit observation masks."""

    first_images: Array
    second_images: Array
    first_valid: Array
    second_valid: Array
    target_forward_rc: Array | None
    target_backward_rc: Array | None
    target_valid: Array | None
    geometry: ImageGeometry2D | None
    scenario_ids: tuple[str, ...] = eqx.field(static=True)
    partition: DatasetPartition = eqx.field(static=True)
    dataset_id: str = eqx.field(static=True)

    def __init__(
        self,
        first_images: ArrayLike,
        second_images: ArrayLike,
        /,
        *,
        first_valid: ArrayLike | None = None,
        second_valid: ArrayLike | None = None,
        target_forward_rc: ArrayLike | None = None,
        target_backward_rc: ArrayLike | None = None,
        target_valid: ArrayLike | None = None,
        geometry: ImageGeometry2D | None = None,
        scenario_ids: Sequence[str] = (),
        partition: DatasetPartition = "training",
        dataset_id: str | None = None,
    ):
        first = jnp.asarray(first_images)
        second = jnp.asarray(second_images)
        if first.ndim != 4 or second.shape != first.shape:
            raise ValueError("Learned PIV images must have equal NHWC batch shape.")
        if first.shape[0] <= 0 or first.shape[-1] <= 0:
            raise ValueError(
                "Learned PIV datasets require at least one case and channel."
            )
        if jnp.issubdtype(first.dtype, jnp.complexfloating) or jnp.issubdtype(
            second.dtype, jnp.complexfloating
        ):
            raise TypeError("Learned PIV images must be real-valued.")
        if not jnp.issubdtype(first.dtype, jnp.inexact):
            first = first.astype(float)
        if not jnp.issubdtype(second.dtype, jnp.inexact):
            second = second.astype(float)
        dtype = jnp.result_type(first.dtype, second.dtype)
        first = first.astype(dtype)
        second = second.astype(dtype)
        batch_size, rows, columns, _ = first.shape
        mask_shape = (batch_size, rows, columns)
        first_mask = (
            jnp.ones(mask_shape, dtype=bool)
            if first_valid is None
            else jnp.asarray(first_valid, dtype=bool)
        )
        second_mask = (
            jnp.ones(mask_shape, dtype=bool)
            if second_valid is None
            else jnp.asarray(second_valid, dtype=bool)
        )
        if first_mask.shape != mask_shape or second_mask.shape != mask_shape:
            raise ValueError(
                "Learned PIV masks must match the image batch and spatial shape."
            )
        first_mask = first_mask & jnp.all(jnp.isfinite(first), axis=-1)
        second_mask = second_mask & jnp.all(jnp.isfinite(second), axis=-1)
        target_shape = mask_shape + (2,)
        forward_target = (
            None
            if target_forward_rc is None
            else jnp.asarray(target_forward_rc, dtype=dtype)
        )
        backward_target = (
            None
            if target_backward_rc is None
            else jnp.asarray(target_backward_rc, dtype=dtype)
        )
        if forward_target is not None and forward_target.shape != target_shape:
            raise ValueError(
                "target_forward_rc must have shape (batch, rows, columns, 2)."
            )
        if backward_target is not None and backward_target.shape != target_shape:
            raise ValueError(
                "target_backward_rc must have shape (batch, rows, columns, 2)."
            )
        if (
            target_valid is not None
            and forward_target is None
            and backward_target is None
        ):
            raise ValueError("target_valid requires at least one displacement target.")
        if forward_target is None and backward_target is None:
            target_mask = None
        else:
            target_mask = (
                jnp.ones(mask_shape, dtype=bool)
                if target_valid is None
                else jnp.asarray(target_valid, dtype=bool)
            )
            if target_mask.shape != mask_shape:
                raise ValueError("target_valid must have shape (batch, rows, columns).")
            if forward_target is not None:
                target_mask = target_mask & jnp.all(jnp.isfinite(forward_target), axis=-1)
            if backward_target is not None:
                target_mask = target_mask & jnp.all(
                    jnp.isfinite(backward_target), axis=-1
                )
        if geometry is not None:
            if not isinstance(geometry, ImageGeometry2D):
                raise TypeError("geometry must be an ImageGeometry2D or None.")
            if geometry.image_shape != (rows, columns):
                raise ValueError("Dataset geometry must match the image spatial shape.")
        if partition not in ("training", "validation", "held-out"):
            raise ValueError("partition must be 'training', 'validation', or 'held-out'.")
        identifiers = (
            tuple(f"case-{index}" for index in range(batch_size))
            if not scenario_ids
            else tuple(str(identifier) for identifier in scenario_ids)
        )
        if len(identifiers) != batch_size or any(
            not identifier for identifier in identifiers
        ):
            raise ValueError(
                "scenario_ids must contain one nonempty identifier per case."
            )
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("scenario_ids must be unique.")

        first = jnp.where(first_mask[..., None], first, 0.0)
        second = jnp.where(second_mask[..., None], second, 0.0)
        if forward_target is not None:
            assert target_mask is not None
            forward_target = jnp.where(target_mask[..., None], forward_target, 0.0)
        if backward_target is not None:
            assert target_mask is not None
            backward_target = jnp.where(target_mask[..., None], backward_target, 0.0)
        content_fingerprint = array_tree_fingerprint(
            {
                "first_images": first,
                "second_images": second,
                "first_valid": first_mask,
                "second_valid": second_mask,
                "target_forward_rc": forward_target,
                "target_backward_rc": backward_target,
                "target_valid": target_mask,
            }
        )["sha256"]
        resolved_dataset_id = (
            canonical_fingerprint(
                {
                    "kind": "learned-piv-dataset",
                    "content_sha256": content_fingerprint,
                    "shape": first.shape,
                    "scenario_ids": identifiers,
                    "partition": partition,
                    "geometry_id": None if geometry is None else geometry.geometry_id,
                }
            )
            if dataset_id is None
            else str(dataset_id)
        )
        if not resolved_dataset_id:
            raise ValueError("dataset_id must be nonempty when provided.")

        self.first_images = first
        self.second_images = second
        self.first_valid = first_mask
        self.second_valid = second_mask
        self.target_forward_rc = forward_target
        self.target_backward_rc = backward_target
        self.target_valid = target_mask
        self.geometry = geometry
        self.scenario_ids = identifiers
        self.partition = partition
        self.dataset_id = resolved_dataset_id

    @property
    def case_count(self) -> int:
        return int(self.first_images.shape[0])

    @property
    def image_shape(self) -> tuple[int, int]:
        return (int(self.first_images.shape[1]), int(self.first_images.shape[2]))

    @property
    def channel_count(self) -> int:
        return int(self.first_images.shape[3])


class LearnedPIVTrainingConfig(StrictModule, NonTrainableState):
    """Deterministic Optax update and masked objective contract."""

    loss: MultiScaleRobustPIVLoss
    maximum_steps: int = eqx.field(static=True)
    batch_size: int = eqx.field(static=True)
    learning_rate: float = eqx.field(static=True)
    maximum_gradient_norm: float = eqx.field(static=True)
    jit: bool = eqx.field(static=True)
    config_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_steps: int,
        batch_size: int,
        learning_rate: float = 1e-3,
        maximum_gradient_norm: float = 1.0,
        loss: MultiScaleRobustPIVLoss | None = None,
        jit: bool = True,
    ):
        steps = int(maximum_steps)
        batch = int(batch_size)
        rate = float(learning_rate)
        gradient_norm = float(maximum_gradient_norm)
        if steps < 0:
            raise ValueError("maximum_steps must be non-negative.")
        if batch <= 0:
            raise ValueError("batch_size must be positive.")
        if not math.isfinite(rate) or rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive.")
        if not math.isfinite(gradient_norm) or gradient_norm <= 0.0:
            raise ValueError("maximum_gradient_norm must be finite and positive.")
        loss_ = MultiScaleRobustPIVLoss() if loss is None else loss
        if not isinstance(loss_, MultiScaleRobustPIVLoss):
            raise TypeError("loss must be a MultiScaleRobustPIVLoss or None.")
        self.loss = loss_
        self.maximum_steps = steps
        self.batch_size = batch
        self.learning_rate = rate
        self.maximum_gradient_norm = gradient_norm
        self.jit = bool(jit)
        self.config_id = canonical_fingerprint(
            {
                "kind": "learned-piv-training-config",
                "maximum_steps": steps,
                "batch_size": batch,
                "learning_rate": rate,
                "maximum_gradient_norm": gradient_norm,
                "loss_id": loss_.loss_id,
                "jit": bool(jit),
            }
        )


class LearnedPIVTrainingEvidence(StrictModule, NonTrainableState):
    """Ordered optimization losses, gradient norms, and final logical cursor."""

    total_loss: Array
    supervised_loss: Array
    photometric_loss: Array
    consistency_loss: Array
    smoothness_loss: Array
    gradient_norm: Array
    progress: TrainingProgress = eqx.field(static=True)
    training_id: str = eqx.field(static=True)


class LearnedPIVFitResult(StrictModule):
    """Selected learned model and reproducible training evidence."""

    model: AbstractDensePIVModel
    evidence: LearnedPIVTrainingEvidence


def _per_case_loss(
    model: AbstractDensePIVModel,
    loss: MultiScaleRobustPIVLoss,
    first: Array,
    second: Array,
    first_valid: Array,
    second_valid: Array,
    target_forward: Array | None,
    target_backward: Array | None,
    target_valid: Array | None,
) -> PIVLossResult:
    forward = model(
        model.plan.prepare(
            first, second, first_valid=first_valid, second_valid=second_valid
        )
    )
    backward = model(
        model.plan.prepare(
            second, first, first_valid=second_valid, second_valid=first_valid
        )
    )
    return loss(
        first,
        second,
        forward.displacement_pyramid_rc,
        backward.displacement_pyramid_rc,
        first_valid=first_valid,
        second_valid=second_valid,
        target_forward_rc=target_forward,
        target_backward_rc=target_backward,
        target_valid=target_valid,
    )


def _aggregate_loss(results: PIVLossResult) -> PIVLossResult:
    valid = jnp.any(results.valid)
    status = jnp.where(
        valid,
        PIV_LOSS_SUCCESS,
        PIV_LOSS_INSUFFICIENT_SUPPORT,
    ).astype(jnp.int32)
    return PIVLossResult(
        total=jnp.mean(results.total),
        supervised=jnp.mean(results.supervised),
        photometric=jnp.mean(results.photometric),
        consistency=jnp.mean(results.consistency),
        smoothness=jnp.mean(results.smoothness),
        supervised_valid_count=jnp.sum(results.supervised_valid_count),
        photometric_valid_count=jnp.sum(results.photometric_valid_count),
        consistency_valid_count=jnp.sum(results.consistency_valid_count),
        smoothness_valid_count=jnp.sum(results.smoothness_valid_count),
        valid=valid,
        status=status,
    )


def _dataset_loss(
    model: AbstractDensePIVModel,
    dataset: LearnedPIVDataset,
    loss: MultiScaleRobustPIVLoss,
    indices: Array,
    /,
) -> PIVLossResult:
    first = dataset.first_images[indices]
    second = dataset.second_images[indices]
    first_valid = dataset.first_valid[indices]
    second_valid = dataset.second_valid[indices]
    target_forward = (
        None if dataset.target_forward_rc is None else dataset.target_forward_rc[indices]
    )
    target_backward = (
        None
        if dataset.target_backward_rc is None
        else dataset.target_backward_rc[indices]
    )
    target_valid = None if dataset.target_valid is None else dataset.target_valid[indices]

    def case_without_targets(first_case, second_case, first_mask, second_mask):
        return _per_case_loss(
            model,
            loss,
            first_case,
            second_case,
            first_mask,
            second_mask,
            None,
            None,
            None,
        )

    if target_forward is None and target_backward is None:
        results = jax.vmap(case_without_targets)(first, second, first_valid, second_valid)
    elif target_forward is not None and target_backward is None:
        if target_valid is None:
            results = jax.vmap(
                lambda a, b, ma, mb, forward: _per_case_loss(
                    model, loss, a, b, ma, mb, forward, None, None
                )
            )(first, second, first_valid, second_valid, target_forward)
        else:
            results = jax.vmap(
                lambda a, b, ma, mb, forward, mt: _per_case_loss(
                    model, loss, a, b, ma, mb, forward, None, mt
                )
            )(first, second, first_valid, second_valid, target_forward, target_valid)
    elif target_forward is None and target_backward is not None:
        if target_valid is None:
            results = jax.vmap(
                lambda a, b, ma, mb, backward: _per_case_loss(
                    model, loss, a, b, ma, mb, None, backward, None
                )
            )(first, second, first_valid, second_valid, target_backward)
        else:
            results = jax.vmap(
                lambda a, b, ma, mb, backward, mt: _per_case_loss(
                    model, loss, a, b, ma, mb, None, backward, mt
                )
            )(first, second, first_valid, second_valid, target_backward, target_valid)
    else:
        assert target_forward is not None and target_backward is not None
        if target_valid is None:
            results = jax.vmap(
                lambda a, b, ma, mb, forward, backward: _per_case_loss(
                    model, loss, a, b, ma, mb, forward, backward, None
                )
            )(first, second, first_valid, second_valid, target_forward, target_backward)
        else:
            results = jax.vmap(
                lambda a, b, ma, mb, forward, backward, mt: _per_case_loss(
                    model, loss, a, b, ma, mb, forward, backward, mt
                )
            )(
                first,
                second,
                first_valid,
                second_valid,
                target_forward,
                target_backward,
                target_valid,
            )
    return _aggregate_loss(results)


def evaluate_learned_piv(
    model: AbstractDensePIVModel,
    dataset: LearnedPIVDataset,
    loss: MultiScaleRobustPIVLoss,
    /,
) -> PIVLossResult:
    """Evaluate every fixed case without changing model state."""

    if not isinstance(model, AbstractDensePIVModel):
        raise TypeError("model must satisfy AbstractDensePIVModel.")
    if not isinstance(dataset, LearnedPIVDataset):
        raise TypeError("dataset must be a LearnedPIVDataset.")
    if not isinstance(loss, MultiScaleRobustPIVLoss):
        raise TypeError("loss must be a MultiScaleRobustPIVLoss.")
    if dataset.image_shape != model.plan.image_shape:
        raise ValueError("Dataset image shape does not match the model plan.")
    if dataset.channel_count != model.plan.input_channels:
        raise ValueError("Dataset channels do not match the model plan.")
    return _dataset_loss(model, dataset, loss, jnp.arange(dataset.case_count))


def fit_learned_piv(
    model: AbstractDensePIVModel,
    dataset: LearnedPIVDataset,
    config: LearnedPIVTrainingConfig,
    /,
    *,
    key: Key[Array, ""] = DOC_KEY0,
    optimizer: optax.GradientTransformation | None = None,
    callbacks: Sequence[TrainingCallback] = (),
) -> LearnedPIVFitResult:
    """Fit with deterministic logical-step sampling and standard Optax updates."""

    if not isinstance(model, AbstractDensePIVModel):
        raise TypeError("model must satisfy AbstractDensePIVModel.")
    if not isinstance(dataset, LearnedPIVDataset):
        raise TypeError("dataset must be a LearnedPIVDataset.")
    if dataset.partition != "training":
        raise ValueError("fit_learned_piv requires a dataset marked 'training'.")
    if not isinstance(config, LearnedPIVTrainingConfig):
        raise TypeError("config must be a LearnedPIVTrainingConfig.")
    if dataset.image_shape != model.plan.image_shape:
        raise ValueError("Dataset image shape does not match the model plan.")
    if dataset.channel_count != model.plan.input_channels:
        raise ValueError("Dataset channels do not match the model plan.")
    image_support = np.any(
        np.asarray(dataset.first_valid & dataset.second_valid),
        axis=(1, 2),
    )
    target_support = (
        np.zeros((dataset.case_count,), dtype=bool)
        if dataset.target_valid is None
        else np.any(np.asarray(dataset.target_valid), axis=(1, 2))
    )
    if not np.all(image_support | target_support):
        raise ValueError(
            "Every learned-PIV training case must contain supported image or target "
            "evidence."
        )

    transformation = (
        optax.chain(
            optax.clip_by_global_norm(config.maximum_gradient_norm),
            optax.adam(config.learning_rate),
        )
        if optimizer is None
        else optimizer
    )
    parameters, non_trainable = partition_trainable(model)
    optimizer_state = transformation.init(parameters)
    control = TrainingController(
        total_steps=config.maximum_steps,
        key=key,
        progress=TrainingProgress(),
        callbacks=callbacks,
    )

    def objective(parameters_: AbstractDensePIVModel, indices: Array):
        current_model = combine_trainable(parameters_, non_trainable)
        result = _dataset_loss(current_model, dataset, config.loss, indices)
        return result.total, result

    value_and_grad = eqx.filter_value_and_grad(objective, has_aux=True)

    def update_step(parameters_, optimizer_state_, indices):
        (value, terms), gradients = value_and_grad(parameters_, indices)
        gradient_norm = optax.tree.norm(gradients)
        updates, next_optimizer_state = transformation.update(
            gradients, optimizer_state_, parameters_
        )
        next_parameters = eqx.apply_updates(parameters_, updates)
        return next_parameters, next_optimizer_state, value, terms, gradient_norm

    compiled_update_step = eqx.filter_jit(update_step) if config.jit else update_step
    total_history: list[Array] = []
    supervised_history: list[Array] = []
    photometric_history: list[Array] = []
    consistency_history: list[Array] = []
    smoothness_history: list[Array] = []
    gradient_history: list[Array] = []

    batch_size = min(config.batch_size, dataset.case_count)
    for step in range(config.maximum_steps):
        if control.stop_requested:
            break
        if batch_size == dataset.case_count:
            indices = jnp.arange(dataset.case_count)
        else:
            indices = jr.choice(
                control.key_for(step, site=0),
                dataset.case_count,
                shape=(batch_size,),
                replace=False,
            )
        parameters, optimizer_state, total, terms, gradient_norm = compiled_update_step(
            parameters, optimizer_state, indices
        )
        total_history.append(total)
        supervised_history.append(terms.supervised)
        photometric_history.append(terms.photometric)
        consistency_history.append(terms.consistency)
        smoothness_history.append(terms.smoothness)
        gradient_history.append(gradient_norm)
        control.complete_update(step + 1)
        control.emit("update", metrics={"loss": total, "gradient_norm": gradient_norm})

    fitted_model = combine_trainable(parameters, non_trainable)
    dtype = dataset.first_images.dtype

    def stack_history(values: list[Array]) -> Array:
        return jnp.stack(values) if values else jnp.zeros((0,), dtype=dtype)

    key_words = tuple(int(word) for word in np.asarray(jr.key_data(key)).reshape(-1))
    training_id = canonical_fingerprint(
        {
            "kind": "learned-piv-training-evidence",
            "architecture_id": model.architecture_id,
            "dataset_id": dataset.dataset_id,
            "config_id": config.config_id,
            "key_data": key_words,
            "completed_steps": control.progress.update_step,
        }
    )
    evidence = LearnedPIVTrainingEvidence(
        total_loss=stack_history(total_history),
        supervised_loss=stack_history(supervised_history),
        photometric_loss=stack_history(photometric_history),
        consistency_loss=stack_history(consistency_history),
        smoothness_loss=stack_history(smoothness_history),
        gradient_norm=stack_history(gradient_history),
        progress=control.progress,
        training_id=training_id,
    )
    return LearnedPIVFitResult(model=fitted_model, evidence=evidence)


__all__ = [
    "LearnedPIVDataset",
    "LearnedPIVFitResult",
    "LearnedPIVTrainingConfig",
    "LearnedPIVTrainingEvidence",
    "evaluate_learned_piv",
    "fit_learned_piv",
]
