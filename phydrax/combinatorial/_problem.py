#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from math import prod
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PyTree

from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._types import CombinatorialFeasibility


class AbstractCombinatorialSpace(StrictModule, NonTrainableState):
    """Fixed feasible set with separate logical decisions and objective features."""

    @property
    @abc.abstractmethod
    def structure_id(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def decision_spec(self, /) -> PyTree[jax.ShapeDtypeStruct]:
        raise NotImplementedError

    @abc.abstractmethod
    def feature_spec(self, /) -> PyTree[jax.ShapeDtypeStruct]:
        raise NotImplementedError

    @abc.abstractmethod
    def encode(self, decision: PyTree[Any], /) -> PyTree[Array]:
        raise NotImplementedError

    @abc.abstractmethod
    def audit(self, decision: PyTree[Any], /) -> CombinatorialFeasibility:
        raise NotImplementedError

    @abc.abstractmethod
    def canonicalize(self, decision: PyTree[Any], /) -> PyTree[Any]:
        """Return the canonical representative of one logical decision."""

        raise NotImplementedError


def _path_name(path: tuple[Any, ...], /) -> str:
    return jax.tree_util.keystr(path) or "<root>"


def _validated_costs(
    costs: PyTree[Any],
    feature_spec: PyTree[jax.ShapeDtypeStruct],
    /,
) -> tuple[PyTree[Array], tuple[int, ...], str, Any]:
    cost_paths, cost_tree = jax.tree_util.tree_flatten_with_path(costs)
    feature_paths, feature_tree = jax.tree_util.tree_flatten_with_path(feature_spec)
    if not feature_paths:
        raise ValueError("Combinatorial feature specifications must be nonempty.")
    if cost_tree != feature_tree:
        raise ValueError(
            "costs and objective features must have identical PyTree structure."
        )

    arrays: list[Array] = []
    batch_shape: tuple[int, ...] | None = None
    dtype: np.dtype[Any] | None = None
    for (cost_path, raw), (feature_path, spec) in zip(
        cost_paths, feature_paths, strict=True
    ):
        if cost_path != feature_path:
            raise ValueError("costs and objective features must have identical paths.")
        if not isinstance(spec, jax.ShapeDtypeStruct):
            raise TypeError(
                f"feature specification {_path_name(feature_path)} is not a ShapeDtypeStruct."
            )
        if isinstance(raw, (str, bytes)):
            raise TypeError("Combinatorial costs must be real floating-point arrays.")
        array = jnp.asarray(raw)
        if not jnp.issubdtype(array.dtype, jnp.floating):
            raise TypeError(
                f"cost leaf {_path_name(cost_path)} must use real floating dtype."
            )
        feature_shape = tuple(int(size) for size in spec.shape)
        if array.ndim < len(feature_shape) or (
            feature_shape
            and tuple(int(size) for size in array.shape[-len(feature_shape) :])
            != feature_shape
        ):
            raise ValueError(
                f"cost leaf {_path_name(cost_path)} must end with feature shape "
                f"{feature_shape}; got {array.shape}."
            )
        leading = (
            tuple(int(size) for size in array.shape[: -len(feature_shape)])
            if feature_shape
            else tuple(int(size) for size in array.shape)
        )
        if batch_shape is None:
            batch_shape = leading
        elif leading != batch_shape:
            raise ValueError("Every combinatorial cost leaf must share one batch shape.")
        current_dtype = np.dtype(array.dtype)
        if dtype is None:
            dtype = current_dtype
        elif current_dtype != dtype:
            raise TypeError(
                "Every combinatorial cost leaf must share one floating dtype."
            )
        arrays.append(array)

    assert batch_shape is not None
    assert dtype is not None
    return cost_tree.unflatten(arrays), batch_shape, str(dtype), cost_tree


class LinearCombinatorialProblem(StrictModule):
    """Linear objective over one fixed combinatorial feasible set."""

    space: AbstractCombinatorialSpace
    costs: PyTree[Array]
    batch_shape: tuple[int, ...] = eqx.field(static=True)
    cost_dtype: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    structure_id: str = eqx.field(static=True)
    cost_tree_definition: Any = eqx.field(static=True)

    def __init__(
        self,
        space: AbstractCombinatorialSpace,
        costs: PyTree[Any],
        /,
        *,
        problem_id: str = "linear-combinatorial-problem",
    ):
        if not isinstance(space, AbstractCombinatorialSpace):
            raise TypeError("space must be an AbstractCombinatorialSpace.")
        identifier = str(problem_id)
        if not identifier:
            raise ValueError("problem_id must be nonempty.")
        arrays, batch_shape, dtype, tree = _validated_costs(
            costs,
            space.feature_spec(),
        )
        self.space = space
        self.costs = arrays
        self.batch_shape = batch_shape
        self.cost_dtype = dtype
        self.problem_id = identifier
        self.structure_id = space.structure_id
        self.cost_tree_definition = tree

    @property
    def batch_size(self) -> int:
        """Return the flattened number of independent cost instances."""

        return prod(self.batch_shape) if self.batch_shape else 1

    def with_costs(self, costs: PyTree[Any], /) -> LinearCombinatorialProblem:
        """Return the same fixed feasible set under compatible numeric costs."""

        refreshed = LinearCombinatorialProblem(
            self.space,
            costs,
            problem_id=self.problem_id,
        )
        if refreshed.batch_shape != self.batch_shape:
            raise ValueError("Refreshed combinatorial costs must preserve batch shape.")
        if refreshed.cost_dtype != self.cost_dtype:
            raise TypeError("Refreshed combinatorial costs must preserve dtype.")
        return refreshed

    def objective(self, features: PyTree[Any], /) -> Array:
        """Evaluate the declared linear objective on batched objective features."""

        feature_spec = self.space.feature_spec()
        cost_leaves, cost_tree = jax.tree_util.tree_flatten(self.costs)
        feature_leaves, feature_tree = jax.tree_util.tree_flatten(features)
        spec_leaves, spec_tree = jax.tree_util.tree_flatten(feature_spec)
        if cost_tree != feature_tree or cost_tree != spec_tree:
            raise ValueError(
                "features must preserve the declared objective PyTree structure."
            )
        value: Array | None = None
        for cost, raw_feature, spec in zip(
            cost_leaves, feature_leaves, spec_leaves, strict=True
        ):
            feature = jnp.asarray(raw_feature, dtype=cost.dtype)
            expected = self.batch_shape + tuple(int(size) for size in spec.shape)
            if feature.shape != expected:
                raise ValueError(
                    f"objective feature shape must be {expected}; got {feature.shape}."
                )
            product_ = cost * feature
            feature_rank = len(spec.shape)
            contribution = (
                jnp.sum(product_, axis=tuple(range(-feature_rank, 0)))
                if feature_rank
                else product_
            )
            value = contribution if value is None else value + contribution
        assert value is not None
        return value


__all__ = [
    "AbstractCombinatorialSpace",
    "LinearCombinatorialProblem",
]
