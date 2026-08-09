#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, PyTree

from .._strict import StrictModule
from ._distributions import AbstractDistribution


class AbstractBijector(StrictModule):
    """Invertible map from unconstrained coordinates to physical parameters."""

    @abstractmethod
    def forward(self, value: ArrayLike, /) -> Array:
        raise NotImplementedError

    @abstractmethod
    def inverse(self, value: ArrayLike, /) -> Array:
        raise NotImplementedError

    @abstractmethod
    def forward_log_det_jacobian(self, value: ArrayLike, /) -> Array:
        raise NotImplementedError


class IdentityBijector(AbstractBijector):
    """Identity map for unconstrained real parameters."""

    def forward(self, value: ArrayLike, /) -> Array:
        return jnp.asarray(value)

    def inverse(self, value: ArrayLike, /) -> Array:
        return jnp.asarray(value)

    def forward_log_det_jacobian(self, value: ArrayLike, /) -> Array:
        return jnp.zeros_like(jnp.asarray(value), dtype=float)


class ExpBijector(AbstractBijector):
    """Exponential map from the real line to positive values."""

    def forward(self, value: ArrayLike, /) -> Array:
        return jnp.exp(jnp.asarray(value))

    def inverse(self, value: ArrayLike, /) -> Array:
        array = jnp.asarray(value)
        if bool(jnp.any(array <= 0.0)):
            raise ValueError("ExpBijector inverse requires strictly positive values.")
        return jnp.log(array)

    def forward_log_det_jacobian(self, value: ArrayLike, /) -> Array:
        return jnp.asarray(value, dtype=float)


class SigmoidIntervalBijector(AbstractBijector):
    """Logistic map from the real line to an open finite interval."""

    lower: Array
    upper: Array

    def __init__(self, lower: ArrayLike, upper: ArrayLike):
        lower_array = jnp.asarray(lower, dtype=float)
        upper_array = jnp.asarray(upper, dtype=float)
        if bool(jnp.any(~jnp.isfinite(lower_array))) or bool(
            jnp.any(~jnp.isfinite(upper_array))
        ):
            raise ValueError("Sigmoid interval bounds must be finite.")
        if bool(jnp.any(upper_array <= lower_array)):
            raise ValueError("Sigmoid interval upper bounds must exceed lower bounds.")
        self.lower = lower_array
        self.upper = upper_array

    def forward(self, value: ArrayLike, /) -> Array:
        value_array = jnp.asarray(value)
        return self.lower + (self.upper - self.lower) * jax.nn.sigmoid(value_array)

    def inverse(self, value: ArrayLike, /) -> Array:
        value_array = jnp.asarray(value)
        if bool(jnp.any((value_array <= self.lower) | (value_array >= self.upper))):
            raise ValueError("SigmoidIntervalBijector inverse requires interior values.")
        probability = (value_array - self.lower) / (self.upper - self.lower)
        return jnp.log(probability) - jnp.log1p(-probability)

    def forward_log_det_jacobian(self, value: ArrayLike, /) -> Array:
        value_array = jnp.asarray(value)
        return (
            jnp.log(self.upper - self.lower)
            + jax.nn.log_sigmoid(value_array)
            + jax.nn.log_sigmoid(-value_array)
        )


_BIJECTOR_LEAF = lambda value: isinstance(value, AbstractBijector)
_PRIOR_LEAF = lambda value: isinstance(value, AbstractDistribution)


class ParameterSpace(StrictModule):
    """Unconstrained posterior coordinates, priors, and physical transformations."""

    initial: PyTree[Any]
    priors: PyTree[AbstractDistribution] | None
    bijectors: PyTree[AbstractBijector]
    custom_log_prior: Callable[[PyTree[Any]], ArrayLike] | None = eqx.field(static=True)

    def __init__(
        self,
        initial: PyTree[Any],
        /,
        *,
        priors: PyTree[AbstractDistribution] | None = None,
        bijectors: PyTree[AbstractBijector] | None = None,
        log_prior: Callable[[PyTree[Any]], ArrayLike] | None = None,
    ):
        leaves = jax.tree_util.tree_leaves(initial)
        if not leaves:
            raise ValueError("ParameterSpace initial position must contain array leaves.")
        for leaf in leaves:
            if not eqx.is_inexact_array(leaf):
                raise TypeError(
                    "Every ParameterSpace position leaf must be an inexact JAX array."
                )
            if bool(jnp.any(~jnp.isfinite(jnp.asarray(leaf)))):
                raise ValueError("ParameterSpace initial position must be finite.")
        if (priors is None) == (log_prior is None):
            raise ValueError("Provide exactly one of priors or log_prior.")
        if log_prior is not None and not callable(log_prior):
            raise TypeError("log_prior must be callable or None.")

        bijector_tree = (
            jax.tree_util.tree_map(lambda _: IdentityBijector(), initial)
            if bijectors is None
            else bijectors
        )
        initial_structure = jax.tree_util.tree_structure(initial)
        bijector_structure = jax.tree_util.tree_structure(
            bijector_tree, is_leaf=_BIJECTOR_LEAF
        )
        if initial_structure != bijector_structure:
            raise ValueError(
                "initial and bijectors must have identical PyTree structure."
            )
        if any(
            not isinstance(value, AbstractBijector)
            for value in jax.tree_util.tree_leaves(bijector_tree, is_leaf=_BIJECTOR_LEAF)
        ):
            raise TypeError("Every bijectors leaf must implement AbstractBijector.")

        if priors is not None:
            prior_structure = jax.tree_util.tree_structure(priors, is_leaf=_PRIOR_LEAF)
            if initial_structure != prior_structure:
                raise ValueError(
                    "initial and priors must have identical PyTree structure."
                )
            if any(
                not isinstance(value, AbstractDistribution)
                for value in jax.tree_util.tree_leaves(priors, is_leaf=_PRIOR_LEAF)
            ):
                raise TypeError("Every priors leaf must implement AbstractDistribution.")

        self.initial = initial
        self.priors = priors
        self.bijectors = bijector_tree
        self.custom_log_prior = log_prior

    def constrain(self, position: PyTree[Any], /) -> PyTree[Any]:
        """Map an unconstrained position to physical parameter values."""
        self._validate_position_structure(position)
        return jax.tree_util.tree_map(
            lambda bijector, value: bijector.forward(value),
            self.bijectors,
            position,
            is_leaf=_BIJECTOR_LEAF,
        )

    def unconstrain(self, physical: PyTree[Any], /) -> PyTree[Any]:
        """Map physical parameter values back to unconstrained coordinates."""
        self._validate_position_structure(physical)
        return jax.tree_util.tree_map(
            lambda bijector, value: bijector.inverse(value),
            self.bijectors,
            physical,
            is_leaf=_BIJECTOR_LEAF,
        )

    def log_abs_det_jacobian(self, position: PyTree[Any], /) -> Array:
        """Return the summed forward log-absolute-determinant Jacobian."""
        self._validate_position_structure(position)
        terms = jax.tree_util.tree_map(
            lambda bijector, value: jnp.sum(bijector.forward_log_det_jacobian(value)),
            self.bijectors,
            position,
            is_leaf=_BIJECTOR_LEAF,
        )
        return _sum_tree(terms)

    def log_prior(self, physical: PyTree[Any], /) -> Array:
        """Evaluate the declared physical-space prior density."""
        self._validate_position_structure(physical)
        if self.custom_log_prior is not None:
            value = jnp.asarray(self.custom_log_prior(physical), dtype=float)
            if value.ndim != 0:
                raise ValueError("Custom log_prior must return a scalar.")
            return value
        if self.priors is None:
            raise RuntimeError("ParameterSpace has no prior specification.")
        terms = jax.tree_util.tree_map(
            lambda prior, value: jnp.sum(prior.log_prob(value)),
            self.priors,
            physical,
            is_leaf=_PRIOR_LEAF,
        )
        return _sum_tree(terms)

    def unconstrained_log_prior(self, position: PyTree[Any], /) -> Array:
        """Evaluate the prior density in unconstrained sampling coordinates."""
        physical = self.constrain(position)
        return self.log_prior(physical) + self.log_abs_det_jacobian(position)

    def sample_prior(
        self,
        key: Array,
        /,
        *,
        num_samples: int,
        constrained: bool = False,
    ) -> PyTree[Array]:
        """Draw independent particles from declared factorized priors."""
        count = int(num_samples)
        if count <= 0:
            raise ValueError("num_samples must be positive.")
        if self.priors is None:
            raise ValueError("Prior sampling requires explicit distribution priors.")
        initial_leaves, treedef = jax.tree_util.tree_flatten(self.initial)
        prior_leaves = jax.tree_util.tree_leaves(
            self.priors,
            is_leaf=_PRIOR_LEAF,
        )
        keys = jr.split(key, len(initial_leaves))
        physical = treedef.unflatten(
            prior.sample(
                sample_key,
                sample_shape=(count,) + tuple(initial.shape),
            )
            for sample_key, prior, initial in zip(
                keys,
                prior_leaves,
                initial_leaves,
            )
        )
        return physical if constrained else self.unconstrain(physical)

    def _validate_position_structure(self, position: PyTree[Any]) -> None:
        if jax.tree_util.tree_structure(position) != jax.tree_util.tree_structure(
            self.initial
        ):
            raise ValueError("Posterior position has incompatible PyTree structure.")


class PosteriorProblem(StrictModule):
    """Deterministic log posterior and optional physical prediction contract."""

    parameter_space: ParameterSpace
    log_likelihood_fn: Callable[[PyTree[Any]], ArrayLike]
    predict_fn: Callable[..., Any] | None = eqx.field(static=True)
    observation_variance_fn: Callable[..., Any] | None = eqx.field(static=True)
    sample_observation_fn: Callable[..., Any] | None = eqx.field(static=True)
    gauss_newton_residual_fn: Callable[[PyTree[Any]], PyTree[Any]] | None = eqx.field(
        static=True
    )

    def __init__(
        self,
        parameter_space: ParameterSpace,
        log_likelihood: Callable[[PyTree[Any]], ArrayLike],
        /,
        *,
        predict: Callable[..., Any] | None = None,
        observation_variance: Callable[..., Any] | None = None,
        sample_observation: Callable[..., Any] | None = None,
        gauss_newton_residual: Callable[[PyTree[Any]], PyTree[Any]] | None = None,
    ):
        if not isinstance(parameter_space, ParameterSpace):
            raise TypeError("parameter_space must be a ParameterSpace.")
        if not callable(log_likelihood):
            raise TypeError("log_likelihood must be callable.")
        if predict is not None and not callable(predict):
            raise TypeError("predict must be callable or None.")
        if observation_variance is not None and not callable(observation_variance):
            raise TypeError("observation_variance must be callable or None.")
        if sample_observation is not None and not callable(sample_observation):
            raise TypeError("sample_observation must be callable or None.")
        if gauss_newton_residual is not None and not callable(gauss_newton_residual):
            raise TypeError("gauss_newton_residual must be callable or None.")
        self.parameter_space = parameter_space
        self.log_likelihood_fn = log_likelihood
        self.predict_fn = predict
        self.observation_variance_fn = observation_variance
        self.sample_observation_fn = sample_observation
        self.gauss_newton_residual_fn = gauss_newton_residual

    @classmethod
    def from_terms(
        cls,
        parameter_space: ParameterSpace,
        terms: Any,
        /,
        *,
        predict: Callable[..., Any] | None = None,
        observation_variance: Callable[..., Any] | None = None,
        sample_observation: Callable[..., Any] | None = None,
        gauss_newton_residual: Callable[[PyTree[Any]], PyTree[Any]] | None = None,
    ) -> PosteriorProblem:
        """Construct a problem by summing explicit normalized likelihood terms."""
        from ._posterior_terms import CompositePosteriorLikelihood

        likelihood = CompositePosteriorLikelihood(terms)
        return cls(
            parameter_space,
            likelihood,
            predict=predict,
            observation_variance=observation_variance,
            sample_observation=sample_observation,
            gauss_newton_residual=gauss_newton_residual,
        )

    @property
    def initial_position(self) -> PyTree[Any]:
        return self.parameter_space.initial

    def log_likelihood(self, physical: PyTree[Any], /) -> Array:
        value = jnp.asarray(self.log_likelihood_fn(physical), dtype=float)
        if value.ndim != 0:
            raise ValueError("log_likelihood must return a scalar.")
        return value

    def log_density(self, position: PyTree[Any], /) -> Array:
        physical = self.parameter_space.constrain(position)
        return (
            self.log_likelihood(physical)
            + self.parameter_space.log_prior(physical)
            + self.parameter_space.log_abs_det_jacobian(position)
        )

    def negative_log_density(self, position: PyTree[Any], /) -> Array:
        return -self.log_density(position)

    def predict(self, position: PyTree[Any], /, *args: Any, **kwargs: Any) -> Any:
        if self.predict_fn is None:
            raise ValueError("PosteriorProblem has no prediction function.")
        physical = self.parameter_space.constrain(position)
        return self.predict_fn(physical, *args, **kwargs)

    def conditional_observation_variance(
        self, position: PyTree[Any], /, *args: Any, **kwargs: Any
    ) -> Any:
        """Evaluate declared measurement variance at one posterior position."""
        if self.observation_variance_fn is None:
            raise ValueError("PosteriorProblem has no observation-variance function.")
        physical = self.parameter_space.constrain(position)
        return self.observation_variance_fn(physical, *args, **kwargs)

    def sample_observation(
        self,
        key: Any,
        position: PyTree[Any],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Draw one measurement realization at one posterior position."""
        if self.sample_observation_fn is None:
            raise ValueError("PosteriorProblem has no observation-sampling function.")
        physical = self.parameter_space.constrain(position)
        return self.sample_observation_fn(key, physical, *args, **kwargs)

    def gauss_newton_residual(self, position: PyTree[Any], /) -> PyTree[Array]:
        """Evaluate explicitly normalized residuals for GGN/Fisher curvature."""
        if self.gauss_newton_residual_fn is None:
            raise ValueError("PosteriorProblem has no Gauss-Newton residual function.")
        physical = self.parameter_space.constrain(position)
        residual = self.gauss_newton_residual_fn(physical)
        if not jax.tree_util.tree_leaves(residual):
            raise ValueError("Gauss-Newton residuals must contain array leaves.")
        return jax.tree_util.tree_map(
            lambda value: jnp.asarray(value, dtype=float),
            residual,
        )

    def validate(self) -> tuple[Array, PyTree[Any]]:
        """Evaluate the initial log density and gradient, rejecting invalid values."""
        value, gradient = jax.value_and_grad(self.log_density)(self.initial_position)
        if value.ndim != 0 or not bool(jnp.isfinite(value)):
            raise FloatingPointError("Initial posterior log density must be finite.")
        invalid = any(
            bool(jnp.any(~jnp.isfinite(jnp.asarray(leaf))))
            for leaf in jax.tree_util.tree_leaves(gradient)
        )
        if invalid:
            raise FloatingPointError("Initial posterior gradient must be finite.")
        return value, gradient


def _sum_tree(tree: PyTree[Any]) -> Array:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.zeros((), dtype=float)
    return sum((jnp.asarray(leaf, dtype=float) for leaf in leaves), jnp.zeros(()))


__all__ = [
    "AbstractBijector",
    "ExpBijector",
    "IdentityBijector",
    "ParameterSpace",
    "PosteriorProblem",
    "SigmoidIntervalBijector",
]
