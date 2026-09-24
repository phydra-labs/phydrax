#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Key

from ..._differentiation import ComponentAuthority, DerivativeRoute, ObjectiveKind
from ..._fingerprint import canonical_fingerprint
from ..._sampling import derive_key, SampleAddress
from ..._trainable import combine_parameters
from ..._training_kernel import (
    KernelObjective,
    OptaxUpdateRule,
    prepare_training_kernel,
    run_training_attempt,
    TrainingKernelSpec,
    TrainingRejectionBudgetError,
)
from ..._training_objective import _ObjectiveContribution
from ._core import AbstractFlowDistribution


_OBJECTIVE_ID = "flow-negative-log-likelihood"
_SPLIT_ADDRESS = SampleAddress("nn.flows.fit", "validation-split", role="split")
_EPOCH_ADDRESS = SampleAddress("nn.flows.fit", "epoch-order", role="epoch")


def _negative_log_likelihood(parameters, model_state, fixed, batch, keys):
    del keys
    model = combine_parameters(parameters, model_state, fixed)
    value = -jnp.mean(model.log_prob(batch))
    return _ObjectiveContribution(value, 1.0), model_state, ()


@eqx.filter_jit
def _validation_loss(flow: AbstractFlowDistribution, samples: Array) -> Array:
    return -jnp.mean(flow.log_prob(samples))


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
    """Fit an unconditional native flow by maximum likelihood.

    Every minibatch is one attempt of the shared training kernel with `MODEL`
    root authority and one data-fit objective. A nonfinite attempt rolls the flow
    and optimizer state back and raises `FloatingPointError`.
    """

    samples = jnp.asarray(data)
    if samples.ndim != 2 or samples.shape[1:] != flow.shape:
        raise ValueError("Flow training data must match the flow event shape.")
    if flow.cond_shape is not None:
        raise ValueError("Unconditional data fitting does not accept conditional flows.")
    count = samples.shape[0]
    validation_count = round(float(validation_fraction) * count)
    training_count = count - validation_count
    if validation_count <= 0 or training_count <= 0:
        raise ValueError("Flow fitting requires nonempty train and validation splits.")
    rate = float(learning_rate)
    kernel = prepare_training_kernel(
        flow,
        (
            KernelObjective(
                objective_id=_OBJECTIVE_ID,
                kind=ObjectiveKind.DATA_FIT,
                route=DerivativeRoute.DIRECT,
                fn=_negative_log_likelihood,
            ),
        ),
        TrainingKernelSpec(
            OptaxUpdateRule(
                optax.adam(rate),
                rule_id=canonical_fingerprint(
                    {"kind": "fit-flow-to-data-adam", "learning_rate": rate.hex()}
                ),
            ),
            context="fit_flow_to_data",
            rejection_budget=0,
        ),
        root_authority=ComponentAuthority.MODEL,
    )
    state = kernel.init(flow, key)
    permutation = jax.random.permutation(derive_key(key, _SPLIT_ADDRESS), count)
    training = samples[permutation[:training_count]]
    validation = samples[permutation[training_count:]]
    best = kernel.tree(state)
    best_validation = float("inf")
    patience = 0
    train_losses: list[Array] = []
    validation_losses: list[Array] = []
    batch_count = max(1, (training_count + int(batch_size) - 1) // int(batch_size))
    for epoch in range(int(max_epochs)):
        order = jax.random.permutation(
            derive_key(key, _EPOCH_ADDRESS, epoch), training_count
        )
        ordered = training[order]
        epoch_loss = jnp.zeros((), dtype=samples.dtype)
        for batch_index in range(batch_count):
            start = batch_index * int(batch_size)
            stop = min(training_count, start + int(batch_size))
            try:
                state, evidence = run_training_attempt(kernel, state, ordered[start:stop])
            except TrainingRejectionBudgetError as error:
                raise FloatingPointError(
                    f"Flow training produced a nonfinite update in epoch {epoch + 1}; "
                    "the flow was rolled back to its last accepted state."
                ) from error
            epoch_loss = epoch_loss + evidence.value.astype(samples.dtype) * (
                (stop - start) / training_count
            )
        model = kernel.tree(state)
        validation_loss = _validation_loss(model, validation)
        train_losses.append(epoch_loss)
        validation_losses.append(validation_loss)
        validation_value = float(validation_loss)
        if validation_value < best_validation:
            best_validation = validation_value
            best = model
            patience = 0
        else:
            patience += 1
            if patience >= int(max_patience):
                break
    return (
        best,
        jnp.stack(train_losses),
        jnp.stack(validation_losses),
    )


__all__ = ["fit_flow_to_data"]
