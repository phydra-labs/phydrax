#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Key

from ..._trainable import (
    combine_parameters,
    partition_parameters,
    require_parameter_roles,
)
from ._core import AbstractFlowDistribution


def fit_flow_to_data(
    key: Key,
    flow: AbstractFlowDistribution,
    data: Array,
    /,
    *,
    learning_rate: float,
    max_epochs: int,
    max_patience: int,
    batch_size: int,
    validation_fraction: float,
) -> tuple[AbstractFlowDistribution, Array, Array]:
    """Fit an unconditional native flow by maximum likelihood."""

    samples = jnp.asarray(data)
    if samples.ndim != 2 or samples.shape[1:] != flow.shape:
        raise ValueError("Flow training data must match the flow event shape.")
    if flow.cond_shape is not None:
        raise ValueError("Unconditional data fitting does not accept conditional flows.")
    require_parameter_roles(flow, context="fit_flow_to_data")
    count = samples.shape[0]
    validation_count = round(float(validation_fraction) * count)
    training_count = count - validation_count
    if validation_count <= 0 or training_count <= 0:
        raise ValueError("Flow fitting requires nonempty train and validation splits.")
    permutation = jax.random.permutation(key, count)
    training = samples[permutation[:training_count]]
    validation = samples[permutation[training_count:]]
    optimizer = optax.adam(float(learning_rate))
    trainable, model_state, fixed = partition_parameters(flow)
    state = optimizer.init(trainable)

    def loss(parameters, batch):
        model = combine_parameters(parameters, model_state, fixed)
        return -jnp.mean(model.log_prob(batch))

    value_and_grad = eqx.filter_value_and_grad(loss)
    best = trainable
    best_validation = float("inf")
    patience = 0
    train_losses: list[Array] = []
    validation_losses: list[Array] = []
    batch_count = max(1, (training_count + int(batch_size) - 1) // int(batch_size))
    for epoch in range(int(max_epochs)):
        epoch_key = jax.random.fold_in(key, epoch + 1)
        order = jax.random.permutation(epoch_key, training_count)
        ordered = training[order]
        epoch_loss = jnp.zeros((), dtype=samples.dtype)
        for batch_index in range(batch_count):
            start = batch_index * int(batch_size)
            stop = min(training_count, start + int(batch_size))
            batch = ordered[start:stop]
            value, gradients = value_and_grad(trainable, batch)
            updates, state = optimizer.update(gradients, state, trainable)
            trainable = eqx.apply_updates(trainable, updates)
            epoch_loss = epoch_loss + value * ((stop - start) / training_count)
        model = combine_parameters(trainable, model_state, fixed)
        validation_loss = -jnp.mean(model.log_prob(validation))
        train_losses.append(epoch_loss)
        validation_losses.append(validation_loss)
        validation_value = float(validation_loss)
        if validation_value < best_validation:
            best_validation = validation_value
            best = trainable
            patience = 0
        else:
            patience += 1
            if patience >= int(max_patience):
                break
    return (
        combine_parameters(best, model_state, fixed),
        jnp.stack(train_losses),
        jnp.stack(validation_losses),
    )


__all__ = ["fit_flow_to_data"]
