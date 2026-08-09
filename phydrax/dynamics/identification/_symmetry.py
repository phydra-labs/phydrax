#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from .._layout import InputLayout, StateLayout
from ._features import AbstractFeatureLibrary, FeatureEvaluation


class LinearTransformedFeatureLibrary(AbstractFeatureLibrary):
    """A declared linear basis change or invariant subspace of a feature library."""

    base: AbstractFeatureLibrary
    transform: Array
    state_layout: StateLayout
    input_layout: InputLayout | None
    feature_names: tuple[str, ...] = eqx.field(static=True)
    library_id: str = eqx.field(static=True)

    def __init__(
        self,
        base: AbstractFeatureLibrary,
        transform: ArrayLike,
        /,
        *,
        feature_names: Sequence[str],
        transform_id: str,
    ):
        if not isinstance(base, AbstractFeatureLibrary):
            raise TypeError("base must be an AbstractFeatureLibrary.")
        matrix = jnp.asarray(transform)
        names = tuple(str(name) for name in feature_names)
        if matrix.ndim != 2 or matrix.shape != (base.num_features, len(names)):
            raise ValueError(
                "transform must have shape (base.num_features, len(feature_names))."
            )
        if not names or any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("feature_names must be non-empty and unique.")
        if not bool(jnp.all(jnp.isfinite(matrix))):
            raise ValueError("transform must be finite.")
        if not isinstance(transform_id, str) or not transform_id:
            raise ValueError("transform_id must be a non-empty string.")
        self.base = base
        self.transform = matrix
        self.state_layout = base.state_layout
        self.input_layout = base.input_layout
        self.feature_names = names
        self.library_id = "linear-feature-transform:" + canonical_fingerprint(
            {
                "base": base.library_id,
                "transform": np.asarray(matrix).tolist(),
                "transform_id": transform_id,
            }
        )

    def evaluate(
        self, states: ArrayLike, inputs: ArrayLike | None = None, /
    ) -> FeatureEvaluation:
        evaluation = self.base.evaluate(states, inputs)
        values = evaluation.values @ self.transform
        valid = evaluation.valid & jnp.all(jnp.isfinite(values), axis=-1)
        return FeatureEvaluation(
            values=jnp.where(valid[..., None], values, 0.0),
            valid=valid,
            feature_names=self.feature_names,
            library_id=self.library_id,
        )


class SymmetryAveragedFeatureLibrary(AbstractFeatureLibrary):
    """Finite-group Reynolds averaging of every feature over declared actions."""

    base: AbstractFeatureLibrary
    state_actions: tuple[Callable[[Array], Array], ...]
    input_actions: tuple[Callable[[Array], Array] | None, ...]
    weights: Array
    state_layout: StateLayout
    input_layout: InputLayout | None
    feature_names: tuple[str, ...] = eqx.field(static=True)
    symmetry_id: str = eqx.field(static=True)
    library_id: str = eqx.field(static=True)

    def __init__(
        self,
        base: AbstractFeatureLibrary,
        state_actions: Sequence[Callable[[Array], Array]],
        /,
        *,
        input_actions: Sequence[Callable[[Array], Array] | None] | None = None,
        weights: Sequence[float] | None = None,
        symmetry_id: str,
    ):
        if not isinstance(base, AbstractFeatureLibrary):
            raise TypeError("base must be an AbstractFeatureLibrary.")
        actions = tuple(state_actions)
        if not actions or any(not callable(action) for action in actions):
            raise TypeError("state_actions must contain at least one callable.")
        if input_actions is None:
            resolved_input_actions = (None,) * len(actions)
        else:
            resolved_input_actions = tuple(input_actions)
            if len(resolved_input_actions) != len(actions) or any(
                action is not None and not callable(action)
                for action in resolved_input_actions
            ):
                raise TypeError(
                    "input_actions must contain one callable or None per state action."
                )
        if weights is None:
            weight_values = np.full((len(actions),), 1.0 / len(actions))
        else:
            weight_values = np.asarray(tuple(weights), dtype=float)
            if weight_values.shape != (len(actions),):
                raise ValueError("weights must contain one entry per group action.")
            if not np.all(np.isfinite(weight_values)) or np.any(weight_values < 0.0):
                raise ValueError("weights must be finite and nonnegative.")
            total = float(np.sum(weight_values))
            if total <= 0.0:
                raise ValueError("weights must have positive total mass.")
            weight_values = weight_values / total
        if not isinstance(symmetry_id, str) or not symmetry_id:
            raise ValueError("symmetry_id must be a non-empty string.")
        self.base = base
        self.state_actions = actions
        self.input_actions = resolved_input_actions
        self.weights = jnp.asarray(weight_values)
        self.state_layout = base.state_layout
        self.input_layout = base.input_layout
        self.feature_names = base.feature_names
        self.symmetry_id = symmetry_id
        self.library_id = "symmetry-average:" + canonical_fingerprint(
            {
                "base": base.library_id,
                "symmetry": symmetry_id,
                "weights": tuple(float(value) for value in weight_values),
            }
        )

    def evaluate(
        self, states: ArrayLike, inputs: ArrayLike | None = None, /
    ) -> FeatureEvaluation:
        state_values = jnp.asarray(states)
        input_values = None if inputs is None else jnp.asarray(inputs)
        evaluations = []
        for state_action, input_action in zip(
            self.state_actions, self.input_actions, strict=True
        ):
            transformed_state = state_action(state_values)
            transformed_input = (
                input_values
                if input_action is None or input_values is None
                else input_action(input_values)
            )
            evaluations.append(self.base.evaluate(transformed_state, transformed_input))
        valid = jnp.all(
            jnp.stack(tuple(evaluation.valid for evaluation in evaluations), axis=-1),
            axis=-1,
        )
        values = jnp.sum(
            jnp.stack(tuple(evaluation.values for evaluation in evaluations), axis=-1)
            * self.weights,
            axis=-1,
        )
        valid = valid & jnp.all(jnp.isfinite(values), axis=-1)
        return FeatureEvaluation(
            values=jnp.where(valid[..., None], values, 0.0),
            valid=valid,
            feature_names=self.feature_names,
            library_id=self.library_id,
        )


__all__ = [
    "LinearTransformedFeatureLibrary",
    "SymmetryAveragedFeatureLibrary",
]
