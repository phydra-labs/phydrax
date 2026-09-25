#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import optax
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._differentiation import ComponentAuthority, DerivativeRoute, ObjectiveKind
from .._strict import StrictModule
from .._trainable import combine_parameters, ParameterOwner
from .._training_kernel import (
    build_training_checkpoint,
    KernelObjective,
    OptaxUpdateRule,
    prepare_training_kernel,
    restore_training_checkpoint,
    run_training_attempt,
    TrainingKernelSpec,
    TrainingKernelState,
    TrainingRejectionBudgetError,
)
from .._training_objective import _ObjectiveContribution
from ..kernels import AbstractPositiveDefiniteKernel
from ._minibatch_posterior import (
    AbstractObservationFactor,
    LikelihoodBatch,
    MinibatchSource,
)
from ._variational import VariationalConfig


class SparseVariationalGaussianState(StrictModule, ParameterOwner):
    """Whitened scalar inducing-point Gaussian variational state.

    Inducing points, whitened mean and lower factor are all PARAMETER.
    """

    inducing_points: Array
    mean: Array
    unconstrained_lower: Array

    def __init__(
        self,
        inducing_points: ArrayLike,
        mean: ArrayLike,
        unconstrained_lower: ArrayLike,
        /,
    ):
        inducing = jnp.asarray(inducing_points)
        mean_array = jnp.asarray(mean)
        lower = jnp.asarray(unconstrained_lower)
        if inducing.ndim != 2 or inducing.shape[0] == 0:
            raise ValueError(
                "inducing_points must have shape (M, input_size) with M > 0."
            )
        count = inducing.shape[0]
        if mean_array.shape != (count,) or lower.shape != (count, count):
            raise ValueError(
                "Whitened mean/lower shapes must align with inducing points."
            )
        if not all(
            jnp.issubdtype(value.dtype, jnp.floating)
            for value in (inducing, mean_array, lower)
        ):
            raise TypeError("Sparse variational GP arrays must be real floating.")
        if any(
            bool(jnp.any(~jnp.isfinite(value))) for value in (inducing, mean_array, lower)
        ):
            raise ValueError("Sparse variational GP arrays must be finite.")
        self.inducing_points = inducing
        self.mean = mean_array
        self.unconstrained_lower = lower

    @classmethod
    def standard_normal(
        cls,
        inducing_points: ArrayLike,
        /,
        *,
        dtype: Any | None = None,
    ) -> SparseVariationalGaussianState:
        inducing = jnp.asarray(inducing_points, dtype=dtype)
        count = inducing.shape[0] if inducing.ndim == 2 else 0
        if count <= 0:
            raise ValueError("inducing_points must be a nonempty matrix.")
        return cls(
            inducing,
            jnp.zeros((count,), dtype=inducing.dtype),
            jnp.zeros((count, count), dtype=inducing.dtype),
        )

    @property
    def scale_tril(self) -> Array:
        diagonal = jnp.exp(jnp.diag(self.unconstrained_lower))
        return jnp.tril(self.unconstrained_lower, -1) + jnp.diag(diagonal)

    @property
    def covariance(self) -> Array:
        lower = self.scale_tril
        return ein.contract("ik,jk->ij", lower, lower)

    @property
    def kl_standard_normal(self) -> Array:
        lower = self.scale_tril
        covariance_trace = jnp.sum(lower**2)
        log_determinant = 2.0 * jnp.sum(jnp.log(jnp.diag(lower)))
        count = self.mean.size
        return 0.5 * (covariance_trace + jnp.sum(self.mean**2) - count - log_determinant)


class SparseVariationalGaussianProcessELBO(StrictModule):
    """Weighted whitened scalar SVGP ELBO with KL counted exactly once."""

    kernel: AbstractPositiveDefiniteKernel
    observation_factor: AbstractObservationFactor
    regularization: Array
    likelihood_samples: int = eqx.field(static=True)

    def __init__(
        self,
        kernel: AbstractPositiveDefiniteKernel,
        observation_factor: AbstractObservationFactor,
        /,
        *,
        regularization: ArrayLike = 0.0,
        likelihood_samples: int = 8,
    ):
        if not isinstance(kernel, AbstractPositiveDefiniteKernel):
            raise TypeError("kernel must be an AbstractPositiveDefiniteKernel.")
        if not isinstance(observation_factor, AbstractObservationFactor):
            raise TypeError(
                "observation_factor must implement AbstractObservationFactor."
            )
        if observation_factor.semantics != "normalized_likelihood":
            raise ValueError("SVGP ELBO requires normalized_likelihood semantics.")
        regularization_array = jnp.asarray(regularization, dtype=jnp.float64).reshape(())
        if not bool(jnp.isfinite(regularization_array)) or bool(
            regularization_array < 0.0
        ):
            raise ValueError("regularization must be finite and nonnegative.")
        samples = int(likelihood_samples)
        if samples <= 0:
            raise ValueError("likelihood_samples must be positive.")
        self.kernel = kernel
        self.observation_factor = observation_factor
        self.regularization = regularization_array
        self.likelihood_samples = samples

    def inducing_factor(self, state: SparseVariationalGaussianState, /) -> Array:
        covariance = self.kernel.matrix(state.inducing_points, state.inducing_points)
        regularized = covariance + self.regularization * jnp.eye(
            covariance.shape[0], dtype=covariance.dtype
        )
        factor = jnp.linalg.cholesky(regularized)
        return eqx.error_if(
            factor,
            jnp.any(~jnp.isfinite(factor)) | jnp.any(jnp.diag(factor) <= 0.0),
            "SVGP inducing covariance is not positive definite under the declared regularization.",
        )

    def latent_moments(
        self,
        state: SparseVariationalGaussianState,
        points: ArrayLike,
        /,
    ) -> tuple[Array, Array]:
        inputs = jnp.asarray(points)
        if inputs.ndim != 2:
            raise ValueError("SVGP inputs must have shape (batch, input_size).")
        factor = self.inducing_factor(state)
        cross = self.kernel.matrix(inputs, state.inducing_points)
        features = jsp.linalg.solve_triangular(factor, cross.T, lower=True).T
        mean = ein.contract("bi,i->b", features, state.mean)
        projected_prior = jnp.sum(features**2, axis=1)
        conditional_variance = self.kernel.diagonal(inputs) - projected_prior
        transformed = ein.contract("bi,ij->bj", features, state.scale_tril)
        variance = conditional_variance + jnp.sum(transformed**2, axis=1)
        return mean, jnp.maximum(variance, 0.0)

    def __call__(
        self,
        state: SparseVariationalGaussianState,
        batch: LikelihoodBatch,
        /,
        *,
        key: Array,
    ) -> Array:
        points = _batch_points(batch)
        mean, variance = self.latent_moments(state, points)
        keys = jr.split(key, self.likelihood_samples)

        def one(sample_key: Array) -> Array:
            latent = mean + jnp.sqrt(variance) * jr.normal(
                sample_key, mean.shape, dtype=mean.dtype
            )
            factors = jnp.asarray(self.observation_factor.log_factors(latent, batch))
            if factors.shape != batch.factor_mask.shape:
                raise ValueError(
                    "Observation factors must align with the batch capacity."
                )
            return jnp.sum(batch.estimator_weights * factors)

        expected_log_likelihood = jnp.mean(jax.vmap(one)(keys))
        return expected_log_likelihood - state.kl_standard_normal

    def predict(
        self,
        state: SparseVariationalGaussianState,
        query_points: ArrayLike,
        /,
    ) -> tuple[Array, Array]:
        return self.latent_moments(state, query_points)


class SparseVariationalGaussianProcessResult(StrictModule):
    """Resumable optimized SVGP state and objective trace.

    `training_state` is the committed training-kernel state (parameters,
    optimizer state, root key, and attempt/accepted cursors) that a continuation
    resumes; `training_checkpoint_id` names the kernel it belongs to.
    """

    state: SparseVariationalGaussianState
    training_state: TrainingKernelState
    objective_trace: Array
    final_step: int = eqx.field(static=True)
    source_fingerprint: str = eqx.field(static=True)
    training_checkpoint_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        state: SparseVariationalGaussianState,
        training_state: TrainingKernelState,
        objective_trace: ArrayLike,
        final_step: int,
        source_fingerprint: str,
        training_checkpoint_id: str,
    ):
        trace = jnp.asarray(objective_trace, dtype=jnp.float64)
        if trace.ndim != 1 or trace.size == 0:
            raise ValueError("objective_trace must be a nonempty vector.")
        if not isinstance(training_state, TrainingKernelState):
            raise TypeError("training_state must be a TrainingKernelState.")
        self.state = state
        self.training_state = training_state
        self.objective_trace = trace
        self.final_step = int(final_step)
        self.source_fingerprint = str(source_fingerprint)
        self.training_checkpoint_id = str(training_checkpoint_id)

    def predict(
        self,
        elbo: SparseVariationalGaussianProcessELBO,
        query_points: ArrayLike,
        /,
    ) -> tuple[Array, Array]:
        return elbo.predict(self.state, query_points)


def _negative_elbo(parameters, model_state, fixed, payload, keys):
    """Negative SVGP ELBO; the payload is `(elbo, batch)`."""
    elbo, batch = payload
    complete = combine_parameters(parameters, model_state, fixed)
    value = -elbo(complete, batch, key=keys.attempt_key("expected-likelihood"))
    return _ObjectiveContribution(value, 1.0), model_state, ()


def fit_sparse_variational_gaussian_process(
    elbo: Any,
    source: MinibatchSource,
    initial_state: SparseVariationalGaussianState,
    /,
    *,
    key: Array,
    config: VariationalConfig | None = None,
    optimizer: optax.GradientTransformation | None = None,
    continuation: SparseVariationalGaussianProcessResult | None = None,
) -> SparseVariationalGaussianProcessResult:
    """Optimize a weighted SVGP ELBO with absolute-step deterministic batches.

    Every step is one attempt of the shared training kernel (`MODEL` root
    authority, one data-fit objective); step `k` consumes batch
    `k % batches_per_epoch` of epoch `k // batches_per_epoch`. The likelihood
    Monte Carlo key is attempt-addressed (`SampleAddress("training",
    "svgp-negative-elbo", target=("expected-likelihood",), role="attempt")`). A
    continuation resumes its committed kernel state, including its root key, so
    `key` seeds fresh runs only. A nonfinite step rolls back and raises
    `FloatingPointError`.
    """
    if not callable(elbo):
        raise TypeError("elbo must be a callable sparse variational GP objective.")
    if not isinstance(initial_state, SparseVariationalGaussianState):
        raise TypeError("initial_state must be SparseVariationalGaussianState.")
    configuration = VariationalConfig() if config is None else config
    if not isinstance(configuration, VariationalConfig):
        raise TypeError("config must be VariationalConfig or None.")
    if optimizer is None:
        transformation = optax.chain(
            optax.clip_by_global_norm(configuration.gradient_clip),
            optax.adam(configuration.learning_rate),
        )
        rule_id = (
            "svgp-clipped-adam:"
            f"{configuration.gradient_clip.hex()}:{configuration.learning_rate.hex()}"
        )
    else:
        transformation = optimizer
        rule_id = "svgp-caller-optimizer"
    if continuation is not None:
        if not isinstance(continuation, SparseVariationalGaussianProcessResult):
            raise TypeError("continuation must be an SVGP result or None.")
        if continuation.source_fingerprint != source.fingerprint:
            raise ValueError("SVGP continuation source fingerprint changed.")
    start = initial_state if continuation is None else continuation.state
    kernel = prepare_training_kernel(
        start,
        (
            KernelObjective(
                objective_id="svgp-negative-elbo",
                kind=ObjectiveKind.DATA_FIT,
                route=DerivativeRoute.DIRECT,
                fn=_negative_elbo,
            ),
        ),
        TrainingKernelSpec(
            OptaxUpdateRule(transformation, rule_id=rule_id),
            context="fit_sparse_variational_gaussian_process",
            rejection_budget=0,
        ),
        root_authority=ComponentAuthority.MODEL,
    )
    if continuation is None:
        state = kernel.init(start, key)
        prior_trace = jnp.empty((0,), dtype=jnp.float64)
    else:
        if continuation.training_checkpoint_id != kernel.checkpoint_id:
            raise ValueError(
                "SVGP continuation belongs to a different training configuration."
            )
        state = restore_training_checkpoint(
            kernel,
            build_training_checkpoint(
                kernel, continuation.training_state, allow_intermediate=True
            ),
        ).state
        prior_trace = continuation.objective_trace
    start_step = int(state.accepted_cursor)
    batches: dict[int, tuple[LikelihoodBatch, ...]] = {}
    recorded: list[Array] = []
    for local_step in range(configuration.num_steps):
        step = start_step + local_step
        epoch = step // source.batches_per_epoch
        batch_index = step % source.batches_per_epoch
        if epoch not in batches:
            batches[epoch] = tuple(source.epoch(epoch))
        try:
            state, evidence = run_training_attempt(
                kernel, state, (elbo, batches[epoch][batch_index])
            )
        except TrainingRejectionBudgetError as error:
            raise FloatingPointError(
                "SVGP optimization produced a nonfinite update; the state was "
                "rolled back to its last accepted step."
            ) from error
        if (local_step + 1) % configuration.record_every == 0 or local_step == 0:
            recorded.append(-evidence.value)
    trace = jnp.concatenate((prior_trace, jnp.stack(recorded)))
    return SparseVariationalGaussianProcessResult(
        state=kernel.tree(state),
        training_state=state,
        objective_trace=trace,
        final_step=start_step + configuration.num_steps,
        source_fingerprint=source.fingerprint,
        training_checkpoint_id=kernel.checkpoint_id,
    )


def _batch_points(batch: LikelihoodBatch, /) -> Array:
    if isinstance(batch.data, dict) and "points" in batch.data:
        points = batch.data["points"]
    elif isinstance(batch.data, tuple) and batch.data:
        points = batch.data[0]
    else:
        points = batch.data
    array = jnp.asarray(points)
    if array.ndim != 2 or array.shape[0] != batch.capacity:
        raise ValueError(
            "SVGP likelihood batches must expose points as data, data[0], or data['points']."
        )
    return array


__all__ = [
    "SparseVariationalGaussianProcessELBO",
    "SparseVariationalGaussianProcessResult",
    "SparseVariationalGaussianState",
    "fit_sparse_variational_gaussian_process",
]
