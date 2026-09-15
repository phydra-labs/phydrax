#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-realization inference adapters for general-relativistic observables."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...observation import (
    CholeskyCovarianceAction,
    ObservationCovarianceAction,
    PrecisionCovarianceAction,
)
from ...uq._foundation import PosteriorRecord
from ..compact_objects._inverse import (
    fixed_branch_model_evaluation,
    FixedBranchInverseAdapter,
    FixedBranchModelEvaluation,
    FixedBranchSensitivityEvidence,
)


class GRInferenceEvaluation(StrictModule):
    """A likelihood value that retains the complete physical forward evidence."""

    forward: FixedBranchModelEvaluation
    residual: Array
    quadratic: Array
    log_likelihood: Array
    log_prior: Array
    log_density: Array
    finite: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    plan_id: str = eqx.field(static=True)


class FixedBranchRayInferencePlan(StrictModule, NonTrainableState):
    """Gaussian inference on a ray realization with frozen event/topology program.

    The adapter must evaluate the already selected ray branch. Expected ray events are
    not reselected here. Event-bearing evaluations remain valid forward calculations,
    but are intentionally ineligible for gradients.
    """

    adapter: FixedBranchInverseAdapter
    observed: Array
    covariance: (
        PrecisionCovarianceAction | CholeskyCovarianceAction | ObservationCovarianceAction
    )
    prior_log_density: Callable[[Array], ArrayLike] | None = eqx.field(static=True)
    observation_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        adapter: FixedBranchInverseAdapter,
        observed: ArrayLike,
        covariance: (
            PrecisionCovarianceAction
            | CholeskyCovarianceAction
            | ObservationCovarianceAction
        ),
        /,
        *,
        observation_id: str,
        prior_log_density: Callable[[Array], ArrayLike] | None = None,
        plan_id: str,
    ):
        if not isinstance(adapter, FixedBranchInverseAdapter):
            raise TypeError("adapter must be a FixedBranchInverseAdapter.")
        if adapter.model_kind != "gr-ray":
            raise ValueError("Ray inference requires a gr-ray inverse adapter.")
        if not isinstance(
            covariance,
            (
                PrecisionCovarianceAction,
                CholeskyCovarianceAction,
                ObservationCovarianceAction,
            ),
        ):
            raise TypeError("covariance must be a native observation covariance action.")
        observation = jax.lax.stop_gradient(jnp.asarray(observed).reshape((-1,)))
        if not eqx.is_inexact_array(observation):
            raise TypeError("Observed GR values must be an inexact array.")
        if observation.shape != (adapter.output_count,):
            raise ValueError("Observed GR values must match the adapter output count.")
        if covariance.layout.size != adapter.output_count:
            raise ValueError("GR observation covariance layout has the wrong size.")
        if bool(jnp.any(~jnp.isfinite(observation))):
            raise ValueError("Observed GR values must be finite.")
        if prior_log_density is not None and not callable(prior_log_density):
            raise TypeError("prior_log_density must be callable or None.")
        observation_identity = _identifier(observation_id, "observation_id")
        declared = _identifier(plan_id, "plan_id")
        self.adapter = adapter
        self.observed = observation
        self.covariance = covariance
        self.prior_log_density = prior_log_density
        self.observation_id = observation_identity
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fixed-branch-gr-ray-inference",
                "declared_id": declared,
                "adapter_id": adapter.adapter_id,
                "observation_id": observation_identity,
                "observed": array_tree_fingerprint(observation),
                "covariance_id": covariance.action_id,
                "prior": None
                if prior_log_density is None
                else type(prior_log_density).__qualname__,
                "prior_arrays": None
                if prior_log_density is None
                else array_tree_fingerprint(prior_log_density),
            }
        )

    def evaluate(self, parameters: ArrayLike, /) -> GRInferenceEvaluation:
        point = jnp.asarray(parameters).reshape((-1,))
        forward = self.adapter.evaluate(point)
        if jnp.issubdtype(forward.values.dtype, jnp.complexfloating):
            raise TypeError(
                "GR inference requires an explicitly realified theory coordinate."
            )
        residual = forward.values - self.observed
        quadratic = self.covariance.quadratic(residual)
        normalization = 0.5 * (
            residual.size * jnp.log(jnp.asarray(2.0 * jnp.pi, dtype=residual.real.dtype))
            + self.covariance.logdet_covariance
        )
        likelihood = -0.5 * quadratic - normalization
        prior = (
            jnp.asarray(0.0, dtype=likelihood.dtype)
            if self.prior_log_density is None
            else jnp.asarray(self.prior_log_density(point), dtype=likelihood.dtype)
        )
        if prior.shape != ():
            raise ValueError("GR prior log density must be scalar.")
        finite = (
            forward.finite
            & jnp.all(jnp.isfinite(residual))
            & jnp.isfinite(quadratic)
            & jnp.isfinite(likelihood)
            & jnp.isfinite(prior)
        )
        converged = finite & forward.converged
        physical = converged & forward.physically_valid
        qualified = physical & forward.qualified
        derivative = qualified & forward.sensitivity_eligible
        log_density = jnp.where(derivative, likelihood + prior, -jnp.inf)
        return GRInferenceEvaluation(
            forward,
            residual,
            quadratic,
            likelihood,
            prior,
            log_density,
            finite,
            converged,
            physical,
            qualified,
            derivative,
            self.plan_id,
        )

    def log_density(self, parameters: ArrayLike, /) -> Array:
        """Return a finite-branch density and refuse nonsmooth branches with ``-inf``."""

        return self.evaluate(parameters).log_density

    def sensitivity(
        self,
        parameters: ArrayLike,
        direction: ArrayLike,
        /,
        *,
        cotangent: ArrayLike | None = None,
        epsilon: float = 1.0e-4,
    ) -> FixedBranchSensitivityEvidence:
        """Delegate JVP/VJP/finite-difference evidence to the frozen ray adapter."""

        return self.adapter.sensitivity(
            parameters,
            direction,
            cotangent=cotangent,
            epsilon=epsilon,
        )


def fixed_branch_ray_inverse_adapter(
    evaluator: Callable[[Array], FixedBranchModelEvaluation],
    expected_branch_signature: ArrayLike,
    /,
    *,
    parameter_count: int,
    output_count: int,
    realization_id: str,
    branch_id: str,
    adapter_id: str,
    evaluator_semantic_id: str | None = None,
    evaluator_numeric_id: str | None = None,
) -> FixedBranchInverseAdapter:
    """Bind one event-free, fixed-topology ray realization for local inference."""

    return FixedBranchInverseAdapter(
        evaluator,
        expected_branch_signature,
        parameter_count=parameter_count,
        output_count=output_count,
        model_kind="gr-ray",
        realization_id=realization_id,
        branch_id=branch_id,
        adapter_id=adapter_id,
        evaluator_semantic_id=evaluator_semantic_id,
        evaluator_numeric_id=evaluator_numeric_id,
    )


def gr_ray_model_evaluation(
    result: Any,
    theory_values: ArrayLike,
    /,
    *,
    realization_id: str,
    branch_id: str,
) -> FixedBranchModelEvaluation:
    """Adapt a ``GRRayResult`` while refusing derivatives through ray events.

    Ray status, event code, and active-mask history jointly form the numeric branch
    signature.  Thus a mask, terminal class, or integration status change invalidates
    the central stencil even if the output happens to remain numerically continuous.
    """

    event_code = result.event_ledger.event_code
    signature = jnp.concatenate(
        (
            jnp.asarray(result.status, dtype=jnp.int32).reshape((-1,)),
            jnp.asarray(event_code, dtype=jnp.int32).reshape((-1,)),
            jnp.asarray(result.active, dtype=jnp.int32).reshape((-1,)),
        )
    )
    event_free = ~jnp.any(result.event_ledger.recorded)
    return fixed_branch_model_evaluation(
        theory_values,
        signature,
        status=result.status_evidence,
        realization_id=realization_id,
        branch_id=branch_id,
        event_free=event_free,
        shock_free=True,
        topology_fixed=True,
    )


class GRPosteriorPrediction(StrictModule, NonTrainableState):
    """Chain/draw-preserving prediction with posterior and forward realization IDs."""

    values: Any
    posterior_id: str = eqx.field(static=True)
    forward_realization_id: str = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)
    chain_count: int = eqx.field(static=True)
    draw_count: int = eqx.field(static=True)
    prediction_id: str = eqx.field(static=True)

    def __init__(
        self,
        values: Any,
        /,
        *,
        posterior_id: str,
        forward_realization_id: str,
        binding_id: str,
        chain_count: int,
        draw_count: int,
    ):
        leaves = [
            leaf for leaf in jax.tree_util.tree_leaves(values) if eqx.is_array(leaf)
        ]
        if not leaves:
            raise ValueError("A GR posterior prediction must contain array leaves.")
        chains = int(chain_count)
        draws = int(draw_count)
        if chains <= 0 or draws <= 0:
            raise ValueError("Posterior chain and draw counts must be positive.")
        expected = (chains, draws)
        if any(leaf.ndim < 2 or leaf.shape[:2] != expected for leaf in leaves):
            raise ValueError(
                "GR prediction changed or collapsed posterior chain/draw axes."
            )
        posterior = _identifier(posterior_id, "posterior_id")
        realization = _identifier(forward_realization_id, "forward_realization_id")
        binding = _identifier(binding_id, "binding_id")
        self.values = values
        self.posterior_id = posterior
        self.chain_count = chains
        self.draw_count = draws
        self.forward_realization_id = realization
        self.binding_id = binding
        self.prediction_id = canonical_fingerprint(
            {
                "kind": "gr-posterior-prediction",
                "posterior_id": posterior,
                "forward_realization_id": realization,
                "binding_id": binding,
                "chain_count": chains,
                "draw_count": draws,
                "values": array_tree_fingerprint(values),
            }
        )


class GRPosteriorRealizationBinding(StrictModule, NonTrainableState):
    """Immutable join between an authoritative posterior and a GR realization."""

    posterior_id: str = eqx.field(static=True)
    inference_plan_id: str = eqx.field(static=True)
    forward_realization_id: str = eqx.field(static=True)
    chain_count: int = eqx.field(static=True)
    draw_count: int = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)

    def __init__(
        self,
        posterior_id: str,
        inference_plan_id: str,
        forward_realization_id: str,
        /,
        *,
        chain_count: int,
        draw_count: int,
    ):
        posterior = _identifier(posterior_id, "posterior_id")
        inference = _identifier(inference_plan_id, "inference_plan_id")
        realization = _identifier(forward_realization_id, "forward_realization_id")
        chains = int(chain_count)
        draws = int(draw_count)
        if chains <= 0 or draws <= 0:
            raise ValueError("Posterior chain and draw counts must be positive.")
        self.posterior_id = posterior
        self.inference_plan_id = inference
        self.forward_realization_id = realization
        self.chain_count = chains
        self.draw_count = draws
        self.binding_id = canonical_fingerprint(
            {
                "kind": "gr-posterior-realization-binding",
                "posterior_id": posterior,
                "inference_plan_id": inference,
                "forward_realization_id": realization,
                "chain_count": chains,
                "draw_count": draws,
            }
        )

    @classmethod
    def from_posterior(
        cls,
        posterior: PosteriorRecord,
        /,
        *,
        inference_plan_id: str,
        forward_realization_id: str,
    ) -> GRPosteriorRealizationBinding:
        if not isinstance(posterior, PosteriorRecord):
            raise TypeError("posterior must be a native PosteriorRecord.")
        return cls(
            posterior.record_id,
            inference_plan_id,
            forward_realization_id,
            chain_count=posterior.result.num_chains,
            draw_count=posterior.result.num_draws,
        )

    def bind_prediction(self, prediction: Any, /) -> GRPosteriorPrediction:
        """Retain native chain/draw axes and attach both authoritative identities."""

        leaves = [
            jnp.asarray(leaf)
            for leaf in jax.tree_util.tree_leaves(prediction)
            if eqx.is_array(leaf)
        ]
        if not leaves:
            raise ValueError("A posterior prediction must contain array leaves.")
        expected = (self.chain_count, self.draw_count)
        if any(leaf.ndim < 2 or leaf.shape[:2] != expected for leaf in leaves):
            raise ValueError(
                "GR prediction changed or collapsed posterior chain/draw axes."
            )
        return GRPosteriorPrediction(
            prediction,
            posterior_id=self.posterior_id,
            forward_realization_id=self.forward_realization_id,
            binding_id=self.binding_id,
            chain_count=self.chain_count,
            draw_count=self.draw_count,
        )

    def predict(
        self,
        posterior: PosteriorRecord,
        /,
        *args: Any,
        **kwargs: Any,
    ) -> GRPosteriorPrediction:
        """Evaluate the authoritative posterior without flattening its sample axes."""

        if not isinstance(posterior, PosteriorRecord):
            raise TypeError("posterior must be a native PosteriorRecord.")
        if posterior.record_id != self.posterior_id:
            raise ValueError(
                "Posterior identity does not match the GR realization binding."
            )
        return self.bind_prediction(posterior.predict(*args, **kwargs))


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")
    return value


__all__ = [
    "FixedBranchRayInferencePlan",
    "GRInferenceEvaluation",
    "GRPosteriorPrediction",
    "GRPosteriorRealizationBinding",
    "fixed_branch_ray_inverse_adapter",
    "gr_ray_model_evaluation",
]
