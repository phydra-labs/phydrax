#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from math import prod
from pathlib import Path
from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from flowjax.distributions import AbstractDistribution
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, PyTree

from .._strict import StrictModule
from ._flow_family import build_default_flow, validate_flow
from ._posterior import PosteriorProblem
from ._variational import (
    AbstractVariationalFamily,
    fit_variational,
    MeanFieldGaussianFamily,
    VariationalConfig,
    VariationalResult,
)


_MEAN_FIELD_TAG = 0
_FLOW_INITIALIZATION_TAG = 1
_FLOW_OPTIMIZATION_TAG = 2


class FlowVariationalFamily(AbstractVariationalFamily):
    """FlowJAX distribution adapted to an unconstrained parameter PyTree."""

    flow: AbstractDistribution
    unravel: Callable[[Array], PyTree[Array]] = eqx.field(static=True)
    tree_definition: Any = eqx.field(static=True)
    event_shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    event_sizes: tuple[int, ...] = eqx.field(static=True)
    dimension: int = eqx.field(static=True)

    def __init__(
        self,
        flow: AbstractDistribution,
        reference: PyTree[Any],
        /,
    ):
        if not isinstance(flow, AbstractDistribution):
            raise TypeError("flow must be a FlowJAX AbstractDistribution.")
        flat_reference, unravel = ravel_pytree(reference)
        if flat_reference.size < 1:
            raise ValueError("Flow variational coordinates cannot be empty.")
        if flow.shape != (int(flat_reference.size),) or flow.cond_shape is not None:
            raise ValueError(
                "Flow event shape must match unconditional parameter coordinates."
            )
        leaves, tree_definition = jax.tree.flatten(reference)
        self.flow = flow
        self.unravel = unravel
        self.tree_definition = tree_definition
        self.event_shapes = tuple(tuple(jnp.asarray(leaf).shape) for leaf in leaves)
        self.event_sizes = tuple(int(jnp.asarray(leaf).size) for leaf in leaves)
        self.dimension = int(flat_reference.size)

    @property
    def family_id(self) -> str:
        return "flowjax-spline"

    def _unflatten_samples(
        self,
        values: Array,
        sample_shape: tuple[int, ...],
        /,
    ) -> PyTree[Array]:
        count = prod(sample_shape) if sample_shape else 1
        matrix = jnp.asarray(values).reshape((count, self.dimension))
        tree = jax.vmap(self.unravel)(matrix)
        if not sample_shape:
            return jax.tree.map(lambda leaf: leaf[0], tree)
        return jax.tree.map(
            lambda leaf: leaf.reshape(sample_shape + leaf.shape[1:]),
            tree,
        )

    def _flatten_samples(self, value: PyTree[Any], /) -> tuple[Array, tuple[int, ...]]:
        if jax.tree.structure(value) != self.tree_definition:
            raise ValueError("value has an incompatible flow parameter PyTree.")
        leaves = jax.tree.leaves(value)
        sample_shape = None
        matrices = []
        for leaf, event_shape, event_size in zip(
            leaves,
            self.event_shapes,
            self.event_sizes,
            strict=True,
        ):
            array = jnp.asarray(leaf)
            if event_shape and (
                array.ndim < len(event_shape)
                or array.shape[-len(event_shape) :] != event_shape
            ):
                raise ValueError("A flow value leaf has an invalid trailing shape.")
            prefix = (
                array.shape[: array.ndim - len(event_shape)]
                if event_shape
                else array.shape
            )
            if sample_shape is None:
                sample_shape = prefix
            elif prefix != sample_shape:
                raise ValueError("Every flow value leaf must share its sample shape.")
            count = prod(prefix) if prefix else 1
            matrices.append(array.reshape((count, event_size)))
        resolved_shape = () if sample_shape is None else tuple(sample_shape)
        return jnp.concatenate(matrices, axis=-1), resolved_shape

    def sample_and_log_prob(
        self,
        key: Array,
        /,
        *,
        sample_shape: tuple[int, ...] = (),
    ) -> tuple[PyTree[Array], Array]:
        shape = tuple(int(size) for size in sample_shape)
        if any(size <= 0 for size in shape):
            raise ValueError("sample_shape dimensions must be positive.")
        values, log_prob = self.flow.sample_and_log_prob(key, sample_shape=shape)
        return self._unflatten_samples(values, shape), log_prob

    def log_prob(self, value: PyTree[Any], /) -> Array:
        matrix, sample_shape = self._flatten_samples(value)
        log_prob = self.flow.log_prob(matrix)
        return log_prob.reshape(sample_shape) if sample_shape else log_prob.reshape(())


class FlowVariationalConfig(StrictModule):
    """Mean-field initialization, flow architecture, and reverse-KL controls."""

    initialization: VariationalConfig
    optimization: VariationalConfig
    initialization_samples: int = eqx.field(static=True)
    flow_layers: int = eqx.field(static=True)
    num_knots: int = eqx.field(static=True)
    nn_width: int = eqx.field(static=True)
    nn_depth: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        initialization: VariationalConfig | None = None,
        optimization: VariationalConfig | None = None,
        initialization_samples: int = 512,
        flow_layers: int = 6,
        num_knots: int = 8,
        nn_width: int = 64,
        nn_depth: int = 2,
    ):
        initialization_ = (
            VariationalConfig(num_steps=500) if initialization is None else initialization
        )
        optimization_ = VariationalConfig() if optimization is None else optimization
        if not isinstance(initialization_, VariationalConfig) or not isinstance(
            optimization_, VariationalConfig
        ):
            raise TypeError("initialization and optimization must be VariationalConfig.")
        counts = tuple(
            int(value)
            for value in (
                initialization_samples,
                flow_layers,
                num_knots,
                nn_width,
                nn_depth,
            )
        )
        if any(value < 1 for value in counts):
            raise ValueError("Flow variational architecture counts must be positive.")
        self.initialization = initialization_
        self.optimization = optimization_
        (
            self.initialization_samples,
            self.flow_layers,
            self.num_knots,
            self.nn_width,
            self.nn_depth,
        ) = counts

    def as_dict(self) -> dict[str, Any]:
        return {
            "initialization": self.initialization.as_dict(),
            "optimization": self.optimization.as_dict(),
            "initialization_samples": self.initialization_samples,
            "flow_layers": self.flow_layers,
            "num_knots": self.num_knots,
            "nn_width": self.nn_width,
            "nn_depth": self.nn_depth,
        }


class FlowVariationalResult(StrictModule):
    """Flow posterior and the mean-field approximation used to initialize it."""

    variational: VariationalResult
    initialization: VariationalResult
    config: FlowVariationalConfig
    approximation_id: str = eqx.field(static=True)

    @property
    def family(self) -> FlowVariationalFamily:
        return cast(FlowVariationalFamily, self.variational.family)

    @property
    def samples(self) -> PyTree[Array]:
        return self.variational.samples

    @property
    def unconstrained_samples(self) -> PyTree[Array]:
        return self.variational.unconstrained_samples

    @property
    def log_target(self) -> Array:
        return self.variational.log_target

    @property
    def log_variational(self) -> Array:
        return self.variational.log_variational

    @property
    def diagnostics(self):
        return self.variational.diagnostics

    @property
    def num_draws(self) -> int:
        return self.variational.num_draws


def fit_flow_variational(
    problem: PosteriorProblem,
    /,
    *,
    key: Array,
    config: FlowVariationalConfig | None = None,
    num_samples: int = 1000,
    initialization: VariationalResult | None = None,
    checkpoint_path: str | Path | None = None,
    checkpoint_every: int | None = None,
    checkpoint_id: str | None = None,
    resume_from: str | Path | None = None,
) -> FlowVariationalResult:
    """Fit an unconditional FlowJAX posterior by reverse KL."""

    if not isinstance(problem, PosteriorProblem):
        raise TypeError("problem must be a PosteriorProblem.")
    config_ = FlowVariationalConfig() if config is None else config
    if not isinstance(config_, FlowVariationalConfig):
        raise TypeError("config must be FlowVariationalConfig or None.")
    if initialization is None:
        initialized = fit_variational(
            problem,
            key=jr.fold_in(key, _MEAN_FIELD_TAG),
            family=MeanFieldGaussianFamily.from_position(problem.initial_position),
            config=config_.initialization,
            num_samples=config_.initialization_samples,
        )
    else:
        if not isinstance(initialization, VariationalResult):
            raise TypeError("initialization must be VariationalResult or None.")
        initialized = initialization
    flat_reference, _ = ravel_pytree(problem.initial_position)
    flat_initialization = jax.vmap(lambda value: ravel_pytree(value)[0])(
        initialized.unconstrained_samples
    )
    if flat_initialization.shape != (
        config_.initialization_samples,
        int(flat_reference.size),
    ):
        raise ValueError(
            "Flow initialization draws do not match initialization_samples and parameter dimension."
        )
    flow = build_default_flow(
        jr.fold_in(key, _FLOW_INITIALIZATION_TAG),
        flat_initialization,
        flow_layers=config_.flow_layers,
        num_knots=config_.num_knots,
        nn_width=config_.nn_width,
        nn_depth=config_.nn_depth,
    )
    validate_flow(
        flow,
        flat_initialization,
        jr.fold_in(key, _FLOW_INITIALIZATION_TAG + 1),
    )
    family = FlowVariationalFamily(flow, problem.initial_position)
    fitted = fit_variational(
        problem,
        key=jr.fold_in(key, _FLOW_OPTIMIZATION_TAG),
        family=family,
        config=config_.optimization,
        num_samples=num_samples,
        checkpoint_path=checkpoint_path,
        checkpoint_every=checkpoint_every,
        checkpoint_id=checkpoint_id,
        resume_from=resume_from,
    )
    return FlowVariationalResult(
        variational=fitted,
        initialization=initialized,
        config=config_,
        approximation_id="reverse-kl/flowjax-spline",
    )


__all__ = [
    "fit_flow_variational",
    "FlowVariationalConfig",
    "FlowVariationalFamily",
    "FlowVariationalResult",
]
