#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Callable, Sequence
from math import comb, floor

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._numerics import (
    dense_index,
    normalize_anisotropy,
    weighted_total_degree_indices,
)
from ..._strict import AbstractAttribute, StrictModule
from .._layout import InputLayout, StateLayout


class FeatureEvaluation(StrictModule):
    """One feature matrix and the samples whose features are finite."""

    values: Array
    valid: Array
    feature_names: tuple[str, ...] = eqx.field(static=True)
    library_id: str = eqx.field(static=True)


class AbstractFeatureLibrary(StrictModule):
    """A fixed ordered dictionary over explicit state and input layouts."""

    state_layout: AbstractAttribute[StateLayout]
    input_layout: AbstractAttribute[InputLayout | None]
    feature_names: AbstractAttribute[tuple[str, ...]]
    library_id: AbstractAttribute[str]

    @property
    def num_features(self) -> int:
        return len(self.feature_names)

    @abc.abstractmethod
    def evaluate(
        self, states: ArrayLike, inputs: ArrayLike | None = None, /
    ) -> FeatureEvaluation:
        raise NotImplementedError

    def __call__(self, states: ArrayLike, inputs: ArrayLike | None = None, /) -> Array:
        return self.evaluate(states, inputs).values


def _batch_variables(
    states: ArrayLike,
    inputs: ArrayLike | None,
    /,
    *,
    state_layout: StateLayout,
    input_layout: InputLayout | None,
) -> tuple[Array, Array]:
    state_values = jnp.asarray(states)
    state_rank = len(state_layout.shape)
    if state_values.ndim < state_rank or (
        state_rank and tuple(state_values.shape[-state_rank:]) != state_layout.shape
    ):
        raise ValueError(
            f"states must end in state layout shape {state_layout.shape}; "
            f"got {state_values.shape}."
        )
    batch_shape = (
        state_values.shape if state_rank == 0 else state_values.shape[:-state_rank]
    )
    flattened_state = state_values.reshape(batch_shape + (state_layout.size,))
    if input_layout is None:
        if inputs is not None:
            raise ValueError("This feature library has no declared input layout.")
        variables = flattened_state
    else:
        if inputs is None:
            raise ValueError("inputs are required by this feature library.")
        input_values = jnp.asarray(inputs)
        expected = batch_shape + input_layout.shape
        if input_values.shape != expected:
            raise ValueError(
                f"inputs must have shape {expected}; got {input_values.shape}."
            )
        flattened_input = input_values.reshape(batch_shape + (input_layout.size,))
        variables = jnp.concatenate((flattened_state, flattened_input), axis=-1)
    valid = jnp.all(jnp.isfinite(variables), axis=-1)
    return variables, valid


def _variable_names(
    state_layout: StateLayout, input_layout: InputLayout | None, /
) -> tuple[str, ...]:
    state_names = tuple(f"state:{name}" for name in state_layout.component_names)
    if input_layout is None:
        return state_names
    return state_names + tuple(f"input:{name}" for name in input_layout.component_names)


def _bounded_weighted_count(
    dimension: int,
    level: int,
    weights: tuple[float, ...],
    *,
    interaction_only: bool,
    limit: int,
) -> int:
    count = 0

    def visit(axis: int, remaining: float) -> None:
        nonlocal count
        if count > limit:
            return
        if axis == dimension:
            count += 1
            return
        maximum = (
            min(1, floor(remaining / weights[axis]))
            if interaction_only
            else floor(remaining / weights[axis])
        )
        for exponent in range(maximum + 1):
            visit(axis + 1, remaining - exponent * weights[axis])

    visit(0, float(level))
    return count


def _monomial_name(exponents: tuple[int, ...], names: tuple[str, ...], /) -> str:
    factors = []
    for name, exponent in zip(names, exponents, strict=True):
        if exponent == 1:
            factors.append(name)
        elif exponent > 1:
            factors.append(f"{name}^{exponent}")
    return "1" if not factors else " * ".join(factors)


class PolynomialFeatureLibrary(AbstractFeatureLibrary):
    """Ordered weighted-total-degree monomials with a pre-allocation size guard."""

    state_layout: StateLayout
    input_layout: InputLayout | None
    exponents: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    feature_names: tuple[str, ...] = eqx.field(static=True)
    library_id: str = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    include_bias: bool = eqx.field(static=True)
    interaction_only: bool = eqx.field(static=True)
    anisotropy: tuple[float, ...] = eqx.field(static=True)

    def __init__(
        self,
        state_layout: StateLayout,
        /,
        *,
        input_layout: InputLayout | None = None,
        degree: int = 2,
        include_bias: bool = True,
        interaction_only: bool = False,
        anisotropy: Sequence[float] | None = None,
        max_features: int = 4096,
    ):
        if not isinstance(state_layout, StateLayout):
            raise TypeError("state_layout must be a StateLayout.")
        if input_layout is not None and not isinstance(input_layout, InputLayout):
            raise TypeError("input_layout must be an InputLayout or None.")
        resolved_degree = int(degree)
        maximum = int(max_features)
        if resolved_degree < 0:
            raise ValueError("degree must be nonnegative.")
        if maximum < 1:
            raise ValueError("max_features must be positive.")
        dimension = state_layout.size + (0 if input_layout is None else input_layout.size)
        weights = normalize_anisotropy(dimension, anisotropy)
        if len(weights) != dimension or any(
            not np.isfinite(value) or value <= 0.0 for value in weights
        ):
            raise ValueError(
                "anisotropy must contain one finite positive weight per variable."
            )
        if anisotropy is None and not interaction_only:
            count = comb(dimension + resolved_degree, resolved_degree)
        else:
            count = _bounded_weighted_count(
                dimension,
                resolved_degree,
                weights,
                interaction_only=bool(interaction_only),
                limit=maximum + int(not include_bias),
            )
        count -= int(not include_bias)
        if count < 1:
            raise ValueError("Polynomial library must contain at least one feature.")
        if count > maximum:
            raise ValueError(
                f"Polynomial library would contain {count} features; "
                f"max_features={maximum}."
            )
        sparse_indices = weighted_total_degree_indices(
            dimension, resolved_degree + 1, weights
        )
        indices = tuple(
            sorted(
                (dense_index(index, dimension) for index in sparse_indices),
                key=lambda exponent: (
                    sum(
                        weight * power
                        for weight, power in zip(weights, exponent, strict=True)
                    ),
                    sum(exponent),
                    tuple(-power for power in exponent),
                ),
            )
        )
        if interaction_only:
            indices = tuple(index for index in indices if max(index, default=0) <= 1)
        if not include_bias:
            indices = tuple(index for index in indices if any(index))
        names = _variable_names(state_layout, input_layout)
        feature_names = tuple(_monomial_name(index, names) for index in indices)
        self.state_layout = state_layout
        self.input_layout = input_layout
        self.exponents = indices
        self.feature_names = feature_names
        self.degree = resolved_degree
        self.include_bias = bool(include_bias)
        self.interaction_only = bool(interaction_only)
        self.anisotropy = weights
        self.library_id = canonical_fingerprint(
            {
                "kind": "polynomial",
                "state_layout": state_layout.layout_id,
                "input_layout": None if input_layout is None else input_layout.layout_id,
                "degree": resolved_degree,
                "include_bias": bool(include_bias),
                "interaction_only": bool(interaction_only),
                "anisotropy": weights,
                "exponents": indices,
            }
        )

    def evaluate(
        self, states: ArrayLike, inputs: ArrayLike | None = None, /
    ) -> FeatureEvaluation:
        variables, source_valid = _batch_variables(
            states,
            inputs,
            state_layout=self.state_layout,
            input_layout=self.input_layout,
        )
        features = jnp.stack(
            tuple(
                jnp.prod(
                    variables ** jnp.asarray(exponent, dtype=jnp.int32),
                    axis=-1,
                )
                for exponent in self.exponents
            ),
            axis=-1,
        )
        valid = source_valid & jnp.all(jnp.isfinite(features), axis=-1)
        return FeatureEvaluation(
            values=jnp.where(valid[..., None], features, 0.0),
            valid=valid,
            feature_names=self.feature_names,
            library_id=self.library_id,
        )


class FourierFeatureLibrary(AbstractFeatureLibrary):
    """Explicit multivariate sine/cosine frequencies in physical variable units."""

    state_layout: StateLayout
    input_layout: InputLayout | None
    frequencies: Array
    phases: Array
    feature_names: tuple[str, ...] = eqx.field(static=True)
    library_id: str = eqx.field(static=True)
    include_bias: bool = eqx.field(static=True)
    include_sine: bool = eqx.field(static=True)
    include_cosine: bool = eqx.field(static=True)

    def __init__(
        self,
        state_layout: StateLayout,
        frequencies: ArrayLike,
        /,
        *,
        input_layout: InputLayout | None = None,
        phases: ArrayLike | None = None,
        include_bias: bool = True,
        include_sine: bool = True,
        include_cosine: bool = True,
        max_features: int = 4096,
    ):
        if not isinstance(state_layout, StateLayout):
            raise TypeError("state_layout must be a StateLayout.")
        if input_layout is not None and not isinstance(input_layout, InputLayout):
            raise TypeError("input_layout must be an InputLayout or None.")
        values = jnp.asarray(frequencies)
        dimension = state_layout.size + (0 if input_layout is None else input_layout.size)
        if values.ndim != 2 or values.shape[1] != dimension:
            raise ValueError(
                f"frequencies must have shape (num_frequencies, {dimension})."
            )
        if values.shape[0] < 1 or bool(jnp.any(~jnp.isfinite(values))):
            raise ValueError("frequencies must be non-empty and finite.")
        phase_values = (
            jnp.zeros((values.shape[0],), dtype=values.dtype)
            if phases is None
            else jnp.asarray(phases, dtype=values.dtype)
        )
        if phase_values.shape != (values.shape[0],) or bool(
            jnp.any(~jnp.isfinite(phase_values))
        ):
            raise ValueError("phases must have one finite scalar per frequency.")
        sine = bool(include_sine)
        cosine = bool(include_cosine)
        bias = bool(include_bias)
        if not sine and not cosine and not bias:
            raise ValueError("At least one Fourier feature must be enabled.")
        count = int(bias) + int(values.shape[0]) * (int(sine) + int(cosine))
        if count > int(max_features):
            raise ValueError(
                f"Fourier library would contain {count} features; "
                f"max_features={int(max_features)}."
            )
        variable_names = _variable_names(state_layout, input_layout)
        mode_names = tuple(
            " + ".join(
                f"{float(coefficient):g}*{name}"
                for coefficient, name in zip(
                    np.asarray(frequency), variable_names, strict=True
                )
                if coefficient != 0.0
            )
            or "0"
            for frequency in np.asarray(values)
        )
        names = ("1",) if bias else ()
        if sine:
            names += tuple(f"sin({name})" for name in mode_names)
        if cosine:
            names += tuple(f"cos({name})" for name in mode_names)
        self.state_layout = state_layout
        self.input_layout = input_layout
        self.frequencies = values
        self.phases = phase_values
        self.feature_names = names
        self.include_bias = bias
        self.include_sine = sine
        self.include_cosine = cosine
        self.library_id = canonical_fingerprint(
            {
                "kind": "fourier",
                "state_layout": state_layout.layout_id,
                "input_layout": None if input_layout is None else input_layout.layout_id,
                "frequencies": np.asarray(values).tolist(),
                "phases": np.asarray(phase_values).tolist(),
                "include_bias": bias,
                "include_sine": sine,
                "include_cosine": cosine,
            }
        )

    def evaluate(
        self, states: ArrayLike, inputs: ArrayLike | None = None, /
    ) -> FeatureEvaluation:
        variables, source_valid = _batch_variables(
            states,
            inputs,
            state_layout=self.state_layout,
            input_layout=self.input_layout,
        )
        angles = oe.contract("...d,kd->...k", variables, self.frequencies) + self.phases
        pieces = []
        if self.include_bias:
            pieces.append(jnp.ones(angles.shape[:-1] + (1,), dtype=angles.dtype))
        if self.include_sine:
            pieces.append(jnp.sin(angles))
        if self.include_cosine:
            pieces.append(jnp.cos(angles))
        features = jnp.concatenate(tuple(pieces), axis=-1)
        valid = source_valid & jnp.all(jnp.isfinite(features), axis=-1)
        return FeatureEvaluation(
            values=jnp.where(valid[..., None], features, 0.0),
            valid=valid,
            feature_names=self.feature_names,
            library_id=self.library_id,
        )


class CustomFeatureLibrary(AbstractFeatureLibrary):
    """User-declared JAX feature map with an explicit output contract and identity."""

    state_layout: StateLayout
    input_layout: InputLayout | None
    function: Callable[[Array, Array | None], Array]
    feature_names: tuple[str, ...] = eqx.field(static=True)
    library_id: str = eqx.field(static=True)

    def __init__(
        self,
        function: Callable[[Array, Array | None], Array],
        /,
        *,
        state_layout: StateLayout,
        feature_names: Sequence[str],
        library_id: str,
        input_layout: InputLayout | None = None,
    ):
        if not callable(function):
            raise TypeError("function must be callable.")
        if not isinstance(state_layout, StateLayout):
            raise TypeError("state_layout must be a StateLayout.")
        if input_layout is not None and not isinstance(input_layout, InputLayout):
            raise TypeError("input_layout must be an InputLayout or None.")
        names = tuple(str(name) for name in feature_names)
        if not names or any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("feature_names must be non-empty and unique.")
        if not isinstance(library_id, str) or not library_id:
            raise ValueError("library_id must be a non-empty string.")
        self.state_layout = state_layout
        self.input_layout = input_layout
        self.function = function
        self.feature_names = names
        self.library_id = library_id

    def evaluate(
        self, states: ArrayLike, inputs: ArrayLike | None = None, /
    ) -> FeatureEvaluation:
        _, source_valid = _batch_variables(
            states,
            inputs,
            state_layout=self.state_layout,
            input_layout=self.input_layout,
        )
        values = jnp.asarray(
            self.function(
                jnp.asarray(states), None if inputs is None else jnp.asarray(inputs)
            )
        )
        expected = source_valid.shape + (self.num_features,)
        if values.shape != expected:
            raise ValueError(
                f"Custom feature map must return shape {expected}; got {values.shape}."
            )
        valid = source_valid & jnp.all(jnp.isfinite(values), axis=-1)
        return FeatureEvaluation(
            values=jnp.where(valid[..., None], values, 0.0),
            valid=valid,
            feature_names=self.feature_names,
            library_id=self.library_id,
        )


def _compatible(libraries: tuple[AbstractFeatureLibrary, ...], owner: str, /) -> None:
    if not libraries:
        raise ValueError(f"{owner} requires at least one feature library.")
    first = libraries[0]
    if any(not isinstance(library, AbstractFeatureLibrary) for library in libraries):
        raise TypeError(f"{owner} entries must be AbstractFeatureLibrary instances.")
    if any(
        library.state_layout.layout_id != first.state_layout.layout_id
        or (None if library.input_layout is None else library.input_layout.layout_id)
        != (None if first.input_layout is None else first.input_layout.layout_id)
        for library in libraries[1:]
    ):
        raise ValueError(f"{owner} libraries must use identical state and input layouts.")


class ConcatenatedFeatureLibrary(AbstractFeatureLibrary):
    """Ordered direct sum of compatible feature dictionaries."""

    libraries: tuple[AbstractFeatureLibrary, ...]
    state_layout: StateLayout
    input_layout: InputLayout | None
    feature_names: tuple[str, ...] = eqx.field(static=True)
    library_id: str = eqx.field(static=True)

    def __init__(self, libraries: Sequence[AbstractFeatureLibrary], /):
        resolved = tuple(libraries)
        _compatible(resolved, "ConcatenatedFeatureLibrary")
        names = tuple(name for library in resolved for name in library.feature_names)
        if len(set(names)) != len(names):
            raise ValueError("Concatenated feature names must be unique.")
        self.libraries = resolved
        self.state_layout = resolved[0].state_layout
        self.input_layout = resolved[0].input_layout
        self.feature_names = names
        self.library_id = canonical_fingerprint(
            {
                "kind": "concatenated",
                "libraries": tuple(library.library_id for library in resolved),
            }
        )

    def evaluate(
        self, states: ArrayLike, inputs: ArrayLike | None = None, /
    ) -> FeatureEvaluation:
        evaluations = tuple(
            library.evaluate(states, inputs) for library in self.libraries
        )
        valid = jnp.all(
            jnp.stack(tuple(item.valid for item in evaluations), axis=-1), axis=-1
        )
        values = jnp.concatenate(tuple(item.values for item in evaluations), axis=-1)
        return FeatureEvaluation(
            values=jnp.where(valid[..., None], values, 0.0),
            valid=valid,
            feature_names=self.feature_names,
            library_id=self.library_id,
        )


class TensorProductFeatureLibrary(AbstractFeatureLibrary):
    """Ordered tensor product of compatible dictionaries with an explicit size guard."""

    left: AbstractFeatureLibrary
    right: AbstractFeatureLibrary
    state_layout: StateLayout
    input_layout: InputLayout | None
    feature_names: tuple[str, ...] = eqx.field(static=True)
    library_id: str = eqx.field(static=True)

    def __init__(
        self,
        left: AbstractFeatureLibrary,
        right: AbstractFeatureLibrary,
        /,
        *,
        max_features: int = 4096,
    ):
        _compatible((left, right), "TensorProductFeatureLibrary")
        count = left.num_features * right.num_features
        if count > int(max_features):
            raise ValueError(
                f"Tensor-product library would contain {count} features; "
                f"max_features={int(max_features)}."
            )
        names = tuple(
            f"({left_name}) * ({right_name})"
            for left_name in left.feature_names
            for right_name in right.feature_names
        )
        self.left = left
        self.right = right
        self.state_layout = left.state_layout
        self.input_layout = left.input_layout
        self.feature_names = names
        self.library_id = canonical_fingerprint(
            {"kind": "tensor-product", "left": left.library_id, "right": right.library_id}
        )

    def evaluate(
        self, states: ArrayLike, inputs: ArrayLike | None = None, /
    ) -> FeatureEvaluation:
        left = self.left.evaluate(states, inputs)
        right = self.right.evaluate(states, inputs)
        products = left.values[..., :, None] * right.values[..., None, :]
        values = products.reshape(left.valid.shape + (self.num_features,))
        valid = left.valid & right.valid & jnp.all(jnp.isfinite(values), axis=-1)
        return FeatureEvaluation(
            values=jnp.where(valid[..., None], values, 0.0),
            valid=valid,
            feature_names=self.feature_names,
            library_id=self.library_id,
        )


__all__ = [
    "AbstractFeatureLibrary",
    "ConcatenatedFeatureLibrary",
    "CustomFeatureLibrary",
    "FeatureEvaluation",
    "FourierFeatureLibrary",
    "PolynomialFeatureLibrary",
    "TensorProductFeatureLibrary",
]
