#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, TYPE_CHECKING

import coordax as cx
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, PyTree

from .._fingerprint import array_tree_fingerprint
from .._frozendict import frozendict
from .._likelihoods import (
    AbstractLikelihood,
    CircularComplexGaussianLikelihood,
    GaussianLikelihood,
)
from .._strict import StrictModule


if TYPE_CHECKING:
    from ..domain import DomainFunction
    from ..terms._likelihood import _AbstractSupervisedLikelihoodTerm
    from ..terms._supervised_dataset import SupervisedDatasetBatch
    from ._gp_actions import AbstractGaussianProcessActionPolicy
    from ._gp_computation_aware import (
        ComputationAwareGaussianProcessDiscrepancy,
        GaussianProcessComputationPolicy,
    )
    from ._gp_scalar import (
        ExactGaussianProcessDiscrepancy,
        SparseGaussianProcessDiscrepancy,
    )

from ._gp_likelihood import GaussianProcessLikelihoodState


class AbstractPosteriorTerm(StrictModule):
    """Deterministic normalized contribution to a posterior log likelihood."""

    label: str = eqx.field(static=True)

    @abstractmethod
    def per_case_log_prob(self, parameters: PyTree[Any], /) -> Array:
        raise NotImplementedError

    def log_prob(self, parameters: PyTree[Any], /) -> Array:
        """Return the sum over independent empirical cases."""
        return jnp.sum(self.per_case_log_prob(parameters)).reshape(())

    def __call__(self, parameters: PyTree[Any], /) -> Array:
        return self.log_prob(parameters)


class FixedObservationLikelihood(AbstractPosteriorTerm):
    """Likelihood for deterministic predictions aligned to fixed observations."""

    target: Array
    likelihood: AbstractLikelihood
    predict_fn: Callable[[PyTree[Any]], ArrayLike | cx.Field] = eqx.field(static=True)
    parameters_fn: Callable[[PyTree[Any]], Mapping[str, ArrayLike | cx.Field]] | None = (
        eqx.field(static=True)
    )

    def __init__(
        self,
        predict: Callable[[PyTree[Any]], ArrayLike | cx.Field],
        target: ArrayLike | cx.Field,
        likelihood: AbstractLikelihood,
        /,
        *,
        parameters: Callable[[PyTree[Any]], Mapping[str, ArrayLike | cx.Field]]
        | None = None,
        label: str = "observation",
    ):
        if not callable(predict):
            raise TypeError("predict must be callable.")
        if not isinstance(likelihood, AbstractLikelihood):
            raise TypeError("likelihood must implement AbstractLikelihood.")
        if parameters is not None and not callable(parameters):
            raise TypeError("parameters must be callable or None.")
        target_array = _field_data(target)
        if target_array.ndim == 0 or int(target_array.shape[0]) <= 0:
            raise ValueError("Fixed observations require a non-empty leading case axis.")
        if not bool(jnp.all(jnp.isfinite(target_array))):
            raise ValueError("Fixed observations must be finite.")
        self.predict_fn = predict
        self.target = target_array
        self.likelihood = likelihood
        self.parameters_fn = parameters
        self.label = _label(label)

    def per_case_log_prob(self, parameters: PyTree[Any], /) -> Array:
        prediction, target = self.likelihood.align_observations(
            _field_data(self.predict_fn(parameters)),
            self.target,
        )
        likelihood_parameters = _likelihood_parameters(self.parameters_fn, parameters)
        values = jnp.asarray(
            self.likelihood.log_prob(prediction, target, **likelihood_parameters)
        )
        _require_real_log_density(values, label=self.label)
        return _reduce_cases(values, int(target.shape[0]), label=self.label)


class FixedResidualLikelihood(AbstractPosteriorTerm):
    """Likelihood for a deterministic residual evaluated on one fixed design."""

    likelihood: AbstractLikelihood
    residual_fn: Callable[[PyTree[Any]], ArrayLike | cx.Field] = eqx.field(static=True)
    target: Array
    parameters_fn: Callable[[PyTree[Any]], Mapping[str, ArrayLike | cx.Field]] | None = (
        eqx.field(static=True)
    )

    def __init__(
        self,
        residual: Callable[[PyTree[Any]], ArrayLike | cx.Field],
        likelihood: AbstractLikelihood,
        /,
        *,
        target: ArrayLike = 0.0,
        parameters: Callable[[PyTree[Any]], Mapping[str, ArrayLike | cx.Field]]
        | None = None,
        label: str = "residual",
    ):
        if not callable(residual):
            raise TypeError("residual must be callable.")
        if not isinstance(likelihood, AbstractLikelihood):
            raise TypeError("likelihood must implement AbstractLikelihood.")
        if parameters is not None and not callable(parameters):
            raise TypeError("parameters must be callable or None.")
        target_array = jnp.asarray(target)
        if not bool(jnp.all(jnp.isfinite(target_array))):
            raise ValueError("Residual targets must be finite.")
        self.residual_fn = residual
        self.likelihood = likelihood
        self.target = target_array
        self.parameters_fn = parameters
        self.label = _label(label)

    def per_case_log_prob(self, parameters: PyTree[Any], /) -> Array:
        residual = _field_data(self.residual_fn(parameters))
        if residual.ndim == 0 or int(residual.shape[0]) <= 0:
            raise ValueError("Fixed residuals require a non-empty leading case axis.")
        target = jnp.broadcast_to(self.target, residual.shape)
        likelihood_parameters = _likelihood_parameters(self.parameters_fn, parameters)
        values = jnp.asarray(
            self.likelihood.log_prob(residual, target, **likelihood_parameters)
        )
        _require_real_log_density(values, label=self.label)
        return _reduce_cases(values, int(residual.shape[0]), label=self.label)


class _ActiveResidual(StrictModule):
    residual_fn: Callable[[PyTree[Any]], Any] = eqx.field(static=True)
    active_indices: Array

    def __init__(
        self,
        residual: Callable[[PyTree[Any]], Any],
        active_indices: ArrayLike,
        /,
    ):
        self.residual_fn = residual
        self.active_indices = jnp.asarray(active_indices, dtype=jnp.int32)

    def __call__(self, parameters: PyTree[Any], /) -> Array:
        value = self.residual_fn(parameters)
        values = value if isinstance(value, tuple) else (value,)
        flattened = tuple(_field_data(item).reshape((-1,)) for item in values)
        return jnp.take(jnp.concatenate(flattened), self.active_indices)


class ResidualPenaltyNoiseModel(StrictModule):
    """Audited normalized noise interpretation of one frozen quadratic penalty."""

    coefficients: Array
    active_indices: Array
    variance: Array
    scale: Array
    penalty_scale: Array
    field: Literal["real", "proper_complex"] = eqx.field(static=True)
    base_measure: str = eqx.field(static=True)
    interpretation_id: str = eqx.field(static=True)
    realization_fingerprint: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        coefficients: ArrayLike,
        penalty_scale: ArrayLike,
        field: Literal["real", "proper_complex"],
        interpretation_id: str,
    ):
        coefficient_array = jnp.asarray(coefficients, dtype=float).reshape((-1,))
        scale_array = jnp.asarray(penalty_scale, dtype=float).reshape(())
        if (
            coefficient_array.size == 0
            or bool(jnp.any(~jnp.isfinite(coefficient_array)))
            or bool(jnp.any(coefficient_array < 0.0))
        ):
            raise ValueError(
                "Quadratic reduction coefficients must be finite and nonnegative."
            )
        if not bool(jnp.isfinite(scale_array)) or not bool(scale_array > 0.0):
            raise ValueError("Residual penalty scale must be finite and positive.")
        if field not in ("real", "proper_complex"):
            raise ValueError("field must be 'real' or 'proper_complex'.")
        identity = str(interpretation_id)
        if not identity:
            raise ValueError("interpretation_id must be non-empty.")
        active = jnp.flatnonzero(coefficient_array > 0.0, size=None)
        if int(active.size) == 0:
            raise ValueError(
                "Residual likelihood interpretation needs a positive coefficient."
            )
        active_coefficients = coefficient_array[active]
        divisor = (
            2.0 * scale_array * active_coefficients
            if field == "real"
            else scale_array * active_coefficients
        )
        variance = 1.0 / divisor
        realization_fingerprint = hashlib.sha256(
            (
                identity
                + ":"
                + field
                + ":"
                + array_tree_fingerprint(
                    {
                        "coefficients": coefficient_array,
                        "penalty_scale": scale_array,
                    }
                )["sha256"]
            ).encode("utf-8")
        ).hexdigest()
        self.coefficients = coefficient_array
        self.active_indices = active.astype(jnp.int32)
        self.variance = variance
        self.scale = jnp.sqrt(variance)
        self.penalty_scale = scale_array
        self.field = field
        self.base_measure = (
            "real_lebesgue" if field == "real" else "real_imaginary_product_lebesgue"
        )
        self.interpretation_id = identity
        self.realization_fingerprint = realization_fingerprint

    @classmethod
    def from_penalty(
        cls,
        term: Any,
        realization: Any,
        /,
        *,
        field: Literal["real", "proper_complex"],
        interpretation_id: str,
    ) -> ResidualPenaltyNoiseModel:
        """Freeze exact integration coefficients without reinterpreting term weights."""
        from ..integration import IntegrationRealization
        from ..terms import ResidualPenalty

        if not isinstance(term, ResidualPenalty):
            raise TypeError("term must be a ResidualPenalty.")
        if not isinstance(realization, IntegrationRealization):
            raise TypeError("realization must be an IntegrationRealization.")
        coefficient_fields = term.quadratic_reduction_coefficients(realization)
        coefficients = jnp.concatenate(
            tuple(jnp.asarray(item.data).reshape((-1,)) for item in coefficient_fields)
        )
        return cls(
            coefficients=coefficients,
            penalty_scale=term.scale,
            field=field,
            interpretation_id=interpretation_id,
        )

    def posterior_term(
        self,
        residual_callable: Callable[[PyTree[Any]], Any],
        /,
        *,
        target: ArrayLike = 0.0,
        label: str | None = None,
    ) -> FixedResidualLikelihood:
        """Build the existing fixed residual adapter on active frozen slots."""
        if not callable(residual_callable):
            raise TypeError("residual_callable must be callable.")
        target_array = jnp.asarray(target)
        if target_array.ndim == 0:
            active_target = jnp.broadcast_to(target_array, self.scale.shape)
        else:
            flat_target = target_array.reshape((-1,))
            if flat_target.shape == self.coefficients.shape:
                active_target = jnp.take(flat_target, self.active_indices)
            elif flat_target.shape == self.scale.shape:
                active_target = flat_target
            else:
                raise ValueError(
                    "Residual target must be scalar, full-capacity, or active-capacity."
                )
        likelihood: AbstractLikelihood = (
            GaussianLikelihood(self.scale)
            if self.field == "real"
            else CircularComplexGaussianLikelihood(self.scale)
        )
        return FixedResidualLikelihood(
            _ActiveResidual(residual_callable, self.active_indices),
            likelihood,
            target=active_target,
            label=label or self.interpretation_id,
        )


class GaussianProcessMarginalLikelihood(AbstractPosteriorTerm):
    """Structured exact or FITC GP marginal likelihood for reusable compilation."""

    discrepancy: ExactGaussianProcessDiscrepancy | SparseGaussianProcessDiscrepancy
    physical_mean_fn: Callable[[PyTree[Any]], ArrayLike | cx.Field] = eqx.field(
        static=True
    )
    state_fn: Callable[[PyTree[Any]], GaussianProcessLikelihoodState] = eqx.field(
        static=True
    )

    def __init__(
        self,
        discrepancy: ExactGaussianProcessDiscrepancy | SparseGaussianProcessDiscrepancy,
        physical_mean: Callable[[PyTree[Any]], ArrayLike | cx.Field],
        /,
        *,
        state: Callable[[PyTree[Any]], GaussianProcessLikelihoodState],
        label: str = "gp_discrepancy",
    ):
        from ._gp_scalar import (
            ExactGaussianProcessDiscrepancy,
            SparseGaussianProcessDiscrepancy,
        )

        if not isinstance(
            discrepancy,
            (ExactGaussianProcessDiscrepancy, SparseGaussianProcessDiscrepancy),
        ):
            raise TypeError(
                "discrepancy must be an exact or sparse scalar GP discrepancy."
            )
        if not callable(physical_mean):
            raise TypeError("physical_mean must be callable.")
        if not callable(state):
            raise TypeError("state must be callable.")
        self.discrepancy = discrepancy
        self.physical_mean_fn = physical_mean
        self.state_fn = state
        self.label = _label(label)

    def per_case_log_prob(self, parameters: PyTree[Any], /) -> Array:
        physical_mean = _field_data(self.physical_mean_fn(parameters))
        state = self.state_fn(parameters)
        if not isinstance(state, GaussianProcessLikelihoodState):
            raise TypeError(
                "GP state callback must return a GaussianProcessLikelihoodState."
            )
        value = self.discrepancy.log_marginal_likelihood(
            physical_mean,
            state=state,
        )
        return jnp.asarray(value, dtype=float).reshape((1,))


class ComputationAwareGaussianProcessELBO(AbstractPosteriorTerm):
    """Full-data variational bound from an action-projected scalar GP posterior."""

    discrepancy: ComputationAwareGaussianProcessDiscrepancy
    physical_mean_fn: Callable[[PyTree[Any]], ArrayLike | cx.Field] = eqx.field(
        static=True
    )
    state_fn: Callable[[PyTree[Any]], GaussianProcessLikelihoodState] = eqx.field(
        static=True
    )
    actions_fn: Callable[[PyTree[Any]], AbstractGaussianProcessActionPolicy] | None = (
        eqx.field(static=True)
    )
    fixed_actions: AbstractGaussianProcessActionPolicy | None
    computation: GaussianProcessComputationPolicy

    def __init__(
        self,
        discrepancy: ComputationAwareGaussianProcessDiscrepancy,
        physical_mean: Callable[[PyTree[Any]], ArrayLike | cx.Field],
        /,
        *,
        state: Callable[[PyTree[Any]], GaussianProcessLikelihoodState],
        actions: AbstractGaussianProcessActionPolicy
        | Callable[[PyTree[Any]], AbstractGaussianProcessActionPolicy],
        computation: GaussianProcessComputationPolicy | None = None,
        label: str = "computation_aware_gp",
    ):
        from ._gp_actions import AbstractGaussianProcessActionPolicy
        from ._gp_computation_aware import (
            ComputationAwareGaussianProcessDiscrepancy,
            GaussianProcessComputationPolicy,
        )

        if not isinstance(discrepancy, ComputationAwareGaussianProcessDiscrepancy):
            raise TypeError(
                "discrepancy must be a ComputationAwareGaussianProcessDiscrepancy."
            )
        if not callable(physical_mean):
            raise TypeError("physical_mean must be callable.")
        if not callable(state):
            raise TypeError("state must be callable.")
        if isinstance(actions, AbstractGaussianProcessActionPolicy):
            fixed_actions = actions
            actions_fn = None
        elif callable(actions):
            fixed_actions = None
            actions_fn = actions
        else:
            raise TypeError(
                "actions must be an AbstractGaussianProcessActionPolicy or callback."
            )
        computation_policy = (
            GaussianProcessComputationPolicy() if computation is None else computation
        )
        if not isinstance(computation_policy, GaussianProcessComputationPolicy):
            raise TypeError("computation must be a GaussianProcessComputationPolicy.")
        self.discrepancy = discrepancy
        self.physical_mean_fn = physical_mean
        self.state_fn = state
        self.actions_fn = actions_fn
        self.fixed_actions = fixed_actions
        self.computation = computation_policy
        self.label = _label(label)

    def per_case_log_prob(self, parameters: PyTree[Any], /) -> Array:
        from ._gp_actions import AbstractGaussianProcessActionPolicy

        physical_mean = _field_data(self.physical_mean_fn(parameters))
        state = self.state_fn(parameters)
        if not isinstance(state, GaussianProcessLikelihoodState):
            raise TypeError(
                "GP state callback must return a GaussianProcessLikelihoodState."
            )
        action_policy = (
            self.fixed_actions if self.actions_fn is None else self.actions_fn(parameters)
        )
        if not isinstance(action_policy, AbstractGaussianProcessActionPolicy):
            raise TypeError(
                "GP actions callback must return an AbstractGaussianProcessActionPolicy."
            )
        assert action_policy is not None
        value = self.discrepancy.elbo(
            physical_mean,
            state=state,
            actions=action_policy,
            computation=self.computation,
        )
        return jnp.asarray(value, dtype=float).reshape((1,))


class FixedSupervisedLikelihood(AbstractPosteriorTerm):
    """Adapter from a supervised likelihood term and its frozen full batch."""

    term: _AbstractSupervisedLikelihoodTerm
    batch: SupervisedDatasetBatch
    functions_fn: Callable[[PyTree[Any]], Mapping[str, DomainFunction]] = eqx.field(
        static=True
    )

    def __init__(
        self,
        term: _AbstractSupervisedLikelihoodTerm,
        functions: Callable[[PyTree[Any]], Mapping[str, DomainFunction]],
        /,
        *,
        batch: SupervisedDatasetBatch | None = None,
        label: str | None = None,
    ):
        from ..terms._likelihood import _AbstractSupervisedLikelihoodTerm
        from ..terms._supervised_dataset import SupervisedDatasetBatch

        if not isinstance(term, _AbstractSupervisedLikelihoodTerm):
            raise TypeError("term must be a supervised likelihood term.")
        if not callable(functions):
            raise TypeError("functions must be callable.")
        batch_value = term.observed_batch() if batch is None else batch
        if not isinstance(batch_value, SupervisedDatasetBatch):
            raise TypeError("batch must be a SupervisedDatasetBatch or None.")
        if int(batch_value.indices.size) <= 0:
            raise ValueError("Fixed supervised batches must be non-empty.")
        self.term = term
        self.functions_fn = functions
        self.batch = batch_value
        self.label = _label(label or term.label or "supervised_likelihood")

    def per_case_log_prob(self, parameters: PyTree[Any], /) -> Array:
        functions = self.functions_fn(parameters)
        if not isinstance(functions, Mapping) or not functions:
            raise TypeError("functions must return a non-empty DomainFunction mapping.")
        return self.term.log_prob(functions, batch=self.batch)


class CompositePosteriorLikelihood(StrictModule):
    """Named deterministic posterior terms evaluated and summed without reweighting."""

    terms: tuple[AbstractPosteriorTerm, ...]
    labels: tuple[str, ...] = eqx.field(static=True)

    def __init__(self, terms: Sequence[AbstractPosteriorTerm], /):
        values = tuple(terms)
        if not values:
            raise ValueError("CompositePosteriorLikelihood requires at least one term.")
        if any(not isinstance(term, AbstractPosteriorTerm) for term in values):
            raise TypeError(
                "Every posterior likelihood term must be an AbstractPosteriorTerm."
            )
        labels = tuple(term.label for term in values)
        if len(set(labels)) != len(labels):
            raise ValueError("Posterior likelihood term labels must be unique.")
        self.terms = values
        self.labels = labels

    def term_values(self, parameters: PyTree[Any], /) -> frozendict[str, Array]:
        """Return each scalar log-likelihood contribution by label."""
        return frozendict(
            {
                label: term.log_prob(parameters)
                for label, term in zip(self.labels, self.terms, strict=True)
            }
        )

    def __call__(self, parameters: PyTree[Any], /) -> Array:
        values = self.term_values(parameters)
        return sum(values.values(), jnp.zeros(())).reshape(())


def _field_data(value: ArrayLike | cx.Field) -> Array:
    return jnp.asarray(value.data if isinstance(value, cx.Field) else value)


def _label(value: str) -> str:
    label = str(value)
    if not label:
        raise ValueError("Posterior likelihood labels must be non-empty.")
    return label


def _likelihood_parameters(
    function: Callable[[PyTree[Any]], Mapping[str, ArrayLike | cx.Field]] | None,
    parameters: PyTree[Any],
) -> dict[str, Array]:
    if function is None:
        return {}
    values = function(parameters)
    if not isinstance(values, Mapping):
        raise TypeError("Likelihood parameters callback must return a mapping.")
    return {str(name): _field_data(value) for name, value in values.items()}


def _require_real_log_density(values: Array, /, *, label: str) -> None:
    if not jnp.issubdtype(values.dtype, jnp.floating):
        raise TypeError(
            f"Posterior term {label!r} must return a real floating log density."
        )


def _reduce_cases(values: Array, case_count: int, /, *, label: str) -> Array:
    if values.ndim == 0 or int(values.shape[0]) != case_count:
        raise ValueError(
            f"Posterior term {label!r} must retain its leading empirical-case axis."
        )
    return values.reshape((case_count, -1)).sum(axis=1)


__all__ = [
    "AbstractPosteriorTerm",
    "CompositePosteriorLikelihood",
    "FixedSupervisedLikelihood",
    "FixedObservationLikelihood",
    "FixedResidualLikelihood",
    "ResidualPenaltyNoiseModel",
    "ComputationAwareGaussianProcessELBO",
    "GaussianProcessMarginalLikelihood",
]
