#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from hashlib import blake2b
from itertools import combinations, combinations_with_replacement
from numbers import Integral, Number
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, ArrayLike

from ..._interpolation import bspline_stencil
from ..._model import AbstractArrayModel
from ..._trainable import NonTrainableState
from ...sparse import EdgeRelation, SparseLinearMap
from .._batch import MLBatch, WeightPolicy
from .._contracts import (
    AbstractRecipe,
    FitResult,
    GradientContract,
    ML_INFEASIBLE,
)
from .._schema import FeatureSchema
from ._common import (
    _align_parameter,
    _check_features,
    _diagnostics,
    _feature_observations,
    _fit_result,
    _weighted_quantiles,
)


class FittedPolynomialFeatures(AbstractArrayModel):
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)
    exponents: Array
    linear_indices: tuple[int, ...] = eqx.field(static=True)
    input_schema: FeatureSchema = eqx.field(static=True)
    output_schema: FeatureSchema = eqx.field(static=True)

    def __init__(
        self,
        exponents: Array,
        /,
        *,
        linear_indices: tuple[int, ...],
        input_schema: FeatureSchema,
        output_schema: FeatureSchema,
    ):
        self.in_size = len(input_schema.names)
        self.out_size = len(output_schema.names)
        self.exponents = jnp.asarray(exponents, dtype=jnp.int32)
        self.linear_indices = tuple(linear_indices)
        self.input_schema = input_schema
        self.output_schema = output_schema

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        values = _check_features(x, self.in_size)
        return jnp.prod(jnp.power(values[..., None, :], self.exponents), axis=-1)

    def transform(self, x: Any, /, *, key: Any = None) -> Array:
        return self(x, key=key)

    def inverse_transform(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        values = _check_features(x, self.out_size)
        if len(self.linear_indices) != self.in_size:
            raise NotImplementedError(
                "Polynomial expansion without every linear term is not invertible."
            )
        return values[..., jnp.asarray(self.linear_indices, dtype=jnp.int32)]


class PolynomialFeatures(AbstractRecipe):
    """Fixed-capacity monomial and interaction expansion."""

    degree: int = eqx.field(static=True)
    interaction_only: bool = eqx.field(static=True)
    include_bias: bool = eqx.field(static=True)
    max_output_features: int = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        degree: int = 2,
        *,
        interaction_only: bool = False,
        include_bias: bool = True,
        max_output_features: int = 4096,
        weight_policy: WeightPolicy = "statistical",
    ):
        if (
            isinstance(degree, bool)
            or not isinstance(degree, Integral)
            or int(degree) < 1
        ):
            raise ValueError("degree must be a positive integer.")
        if int(max_output_features) <= 0:
            raise ValueError("max_output_features must be positive.")
        if weight_policy not in ("none", "statistical", "measure", "product"):
            raise ValueError("Unsupported weight policy.")
        self.degree = int(degree)
        self.interaction_only = bool(interaction_only)
        self.include_bias = bool(include_bias)
        self.max_output_features = int(max_output_features)
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        _x, _weights, mass, effective, valid, status = _feature_observations(
            batch, weight_policy=self.weight_policy
        )
        rows: list[tuple[int, ...]] = []
        names: list[str] = []
        if self.include_bias:
            rows.append((0,) * batch.feature_count)
            names.append("1")
        chooser = combinations if self.interaction_only else combinations_with_replacement
        for total_degree in range(1, self.degree + 1):
            for terms in chooser(range(batch.feature_count), total_degree):
                exponent = tuple(
                    terms.count(index) for index in range(batch.feature_count)
                )
                rows.append(exponent)
                pieces = []
                for name, power in zip(batch.feature_schema.names, exponent, strict=True):
                    if power == 1:
                        pieces.append(name)
                    elif power > 1:
                        pieces.append(f"{name}^{power}")
                names.append("*".join(pieces))
        if len(rows) > self.max_output_features:
            raise ValueError(
                f"Polynomial expansion requires {len(rows)} outputs, exceeding "
                f"max_output_features={self.max_output_features}."
            )
        exponents = jnp.asarray(rows, dtype=jnp.int32)
        linear_indices = []
        for feature in range(batch.feature_count):
            expected = tuple(
                1 if index == feature else 0 for index in range(batch.feature_count)
            )
            linear_indices.append(rows.index(expected))
        output_schema = FeatureSchema(
            tuple(names),
            kinds=("continuous",) * len(names),
            layout_id=batch.feature_schema.layout_id,
        )
        model = FittedPolynomialFeatures(
            exponents,
            linear_indices=tuple(linear_indices),
            input_schema=batch.feature_schema,
            output_schema=output_schema,
        )
        diagnostics = _diagnostics(
            batch,
            output_schema,
            mass,
            effective,
            valid,
            status,
            method="polynomial_features",
            details=(
                ("degree", self.degree),
                ("interaction_only", self.interaction_only),
            ),
        )
        return _fit_result(
            model,
            diagnostics,
            GradientContract(
                prediction_inputs="smooth",
                prediction_parameters="none",
                fit_mode="direct",
            ),
        )


class FittedSplineTransformer(AbstractArrayModel):
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)
    knots: Array
    degree: int = eqx.field(static=True)
    n_basis: int = eqx.field(static=True)
    bounds: Literal["clip", "error"] = eqx.field(static=True)
    input_schema: FeatureSchema = eqx.field(static=True)
    output_schema: FeatureSchema = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        knots: Array,
        /,
        *,
        degree: int,
        n_basis: int,
        bounds: Literal["clip", "error"],
        input_schema: FeatureSchema,
        output_schema: FeatureSchema,
        case_shape: tuple[int, ...],
    ):
        self.in_size = len(input_schema.names)
        self.out_size = len(output_schema.names)
        self.knots = jnp.asarray(knots)
        self.degree = int(degree)
        self.n_basis = int(n_basis)
        self.bounds = bounds
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.case_shape = tuple(case_shape)

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        values = _check_features(x, self.in_size)
        if jnp.issubdtype(values.dtype, jnp.complexfloating):
            raise TypeError("SplineTransformer requires real-valued inputs.")
        knots = _align_parameter(self.knots, values, self.case_shape, trailing_rank=2)
        knots = jnp.broadcast_to(knots, values.shape + (self.knots.shape[-1],))
        flat_values = values.reshape((-1,))
        flat_knots = knots.reshape((-1, knots.shape[-1]))

        def basis_row(knot_row, query):
            stencil = bspline_stencil(
                knot_row,
                query,
                degree=self.degree,
                bounds=self.bounds,
            )
            return (
                jnp.zeros((self.n_basis,), dtype=stencil.weights.dtype)
                .at[stencil.indices]
                .add(stencil.weights)
            )

        basis = jax.vmap(basis_row)(flat_knots, flat_values)
        return basis.reshape(values.shape[:-1] + (self.out_size,))

    def transform(self, x: Any, /, *, key: Any = None) -> Array:
        return self(x, key=key)

    def inverse_transform(self, x: Any, /, *, key: Any = None) -> Array:
        del x, key
        raise NotImplementedError(
            "Spline basis expansion has no single-valued inverse transform."
        )


class SplineTransformer(AbstractRecipe):
    """Per-feature clamped B-spline basis with weighted uniform or quantile knots."""

    n_knots: int = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    knots: Literal["uniform", "quantile"] = eqx.field(static=True)
    bounds: Literal["clip", "error"] = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        n_knots: int = 5,
        degree: int = 3,
        *,
        knots: Literal["uniform", "quantile"] = "uniform",
        bounds: Literal["clip", "error"] = "error",
        weight_policy: WeightPolicy = "statistical",
    ):
        if int(n_knots) < 2:
            raise ValueError("n_knots must be at least two.")
        if int(degree) < 0:
            raise ValueError("degree must be nonnegative.")
        if knots not in ("uniform", "quantile"):
            raise ValueError("knots must be 'uniform' or 'quantile'.")
        if bounds not in ("clip", "error"):
            raise ValueError("bounds must be 'clip' or 'error'.")
        if weight_policy not in ("none", "statistical", "measure", "product"):
            raise ValueError("Unsupported weight policy.")
        self.n_knots = int(n_knots)
        self.degree = int(degree)
        self.knots = knots
        self.bounds = bounds
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        x, weights, mass, effective, valid, status = _feature_observations(
            batch, weight_policy=self.weight_policy
        )
        if jnp.issubdtype(x.dtype, jnp.complexfloating):
            raise TypeError("SplineTransformer requires real-valued features.")
        probabilities = jnp.linspace(0.0, 1.0, self.n_knots, dtype=weights.dtype)
        quantiles = _weighted_quantiles(x, weights, probabilities)
        lower, upper = quantiles[..., 0], quantiles[..., -1]
        constant = upper <= lower
        radius = jnp.sqrt(jnp.finfo(jnp.result_type(x, float)).eps) * jnp.maximum(
            jnp.abs(lower), 1.0
        )
        safe_lower = jnp.where(constant, lower - radius, lower)
        safe_upper = jnp.where(constant, upper + radius, upper)
        if self.knots == "uniform":
            fraction = jnp.linspace(0.0, 1.0, self.n_knots, dtype=weights.dtype)
            base_knots = (
                safe_lower[..., None] + (safe_upper - safe_lower)[..., None] * fraction
            )
        else:
            uniform = (
                safe_lower[..., None]
                + (safe_upper - safe_lower)[..., None] * probabilities
            )
            base_knots = jnp.where(constant[..., None], uniform, quantiles)
        full_knots = jnp.concatenate(
            (
                jnp.repeat(base_knots[..., :1], self.degree, axis=-1),
                base_knots,
                jnp.repeat(base_knots[..., -1:], self.degree, axis=-1),
            ),
            axis=-1,
        )
        n_basis = self.n_knots + self.degree - 1
        names = tuple(
            f"{name}_spline_{basis}"
            for name in batch.feature_schema.names
            for basis in range(n_basis)
        )
        output_schema = FeatureSchema(
            names,
            kinds=("continuous",) * len(names),
            layout_id=batch.feature_schema.layout_id,
        )
        model = FittedSplineTransformer(
            full_knots,
            degree=self.degree,
            n_basis=n_basis,
            bounds=self.bounds,
            input_schema=batch.feature_schema,
            output_schema=output_schema,
            case_shape=batch.case_shape,
        )
        diagnostics = _diagnostics(
            batch,
            output_schema,
            mass,
            effective,
            valid,
            status,
            method="spline_transformer",
            constant=constant,
            details=(
                ("n_knots", self.n_knots),
                ("degree", self.degree),
                ("knots", self.knots),
            ),
        )
        return _fit_result(
            model,
            diagnostics,
            GradientContract(
                prediction_inputs="almost-everywhere",
                prediction_parameters="conditional",
                fit_features="none",
                fit_weights="none",
                fit_mode="stopped",
                nondifferentiable_outputs=("knot_spans",),
                conditions=("Fitted knot order and active spans are held fixed.",),
            ),
        )


class FittedFourierFeatures(AbstractArrayModel):
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)
    origin: Array
    period: Array
    n_frequencies: int = eqx.field(static=True)
    include_bias: bool = eqx.field(static=True)
    include_original: bool = eqx.field(static=True)
    input_schema: FeatureSchema = eqx.field(static=True)
    output_schema: FeatureSchema = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        origin: Array,
        period: Array,
        /,
        *,
        n_frequencies: int,
        include_bias: bool,
        include_original: bool,
        input_schema: FeatureSchema,
        output_schema: FeatureSchema,
        case_shape: tuple[int, ...],
    ):
        self.in_size = len(input_schema.names)
        self.out_size = len(output_schema.names)
        self.origin = jnp.asarray(origin)
        self.period = jnp.asarray(period)
        self.n_frequencies = int(n_frequencies)
        self.include_bias = bool(include_bias)
        self.include_original = bool(include_original)
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.case_shape = tuple(case_shape)

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        values = _check_features(x, self.in_size)
        origin = _align_parameter(self.origin, values, self.case_shape)
        period = _align_parameter(self.period, values, self.case_shape)
        harmonics = jnp.arange(1, self.n_frequencies + 1, dtype=values.real.dtype)
        phase = (
            (2.0 * jnp.pi)
            * (values - origin)[..., :, None]
            * harmonics
            / period[..., :, None]
        )
        pieces = []
        if self.include_original:
            pieces.append(values)
        if self.include_bias:
            pieces.append(jnp.ones(values.shape[:-1] + (1,), dtype=values.dtype))
        pieces.extend(
            (
                jnp.sin(phase).reshape(values.shape[:-1] + (-1,)),
                jnp.cos(phase).reshape(values.shape[:-1] + (-1,)),
            )
        )
        return jnp.concatenate(pieces, axis=-1)

    def transform(self, x: Any, /, *, key: Any = None) -> Array:
        return self(x, key=key)

    def inverse_transform(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        values = _check_features(x, self.out_size)
        if not self.include_original:
            raise NotImplementedError(
                "Periodic Fourier features are not injective without original inputs."
            )
        return values[..., : self.in_size]


class FourierFeatures(AbstractRecipe):
    """Deterministic per-feature harmonic expansion with fitted or explicit periods."""

    n_frequencies: int = eqx.field(static=True)
    period: tuple[float, ...] | None = eqx.field(static=True)
    origin: tuple[float, ...] | None = eqx.field(static=True)
    include_bias: bool = eqx.field(static=True)
    include_original: bool = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        n_frequencies: int = 4,
        *,
        period: Number | Sequence[Number] | None = None,
        origin: Number | Sequence[Number] | None = None,
        include_bias: bool = False,
        include_original: bool = False,
        weight_policy: WeightPolicy = "statistical",
    ):
        if int(n_frequencies) <= 0:
            raise ValueError("n_frequencies must be positive.")
        if weight_policy not in ("none", "statistical", "measure", "product"):
            raise ValueError("Unsupported weight policy.")

        def normalize(value):
            if value is None:
                return None
            raw = (value,) if isinstance(value, Number) else tuple(value)
            converted = tuple(float(item) for item in raw)
            if any(not jnp.isfinite(item) for item in converted):
                raise ValueError("Fourier origins and periods must be finite.")
            return converted

        period_ = normalize(period)
        if period_ is not None and any(item <= 0.0 for item in period_):
            raise ValueError("Fourier periods must be positive.")
        self.n_frequencies = int(n_frequencies)
        self.period = period_
        self.origin = normalize(origin)
        self.include_bias = bool(include_bias)
        self.include_original = bool(include_original)
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        x, weights, mass, effective, valid, status = _feature_observations(
            batch, weight_policy=self.weight_policy
        )
        if jnp.issubdtype(x.dtype, jnp.complexfloating) and (
            self.period is None or self.origin is None
        ):
            raise TypeError(
                "Complex Fourier features require explicit real periods and origins."
            )
        real_x = x.real.astype(jnp.result_type(x.real, float))
        minimum = jnp.min(jnp.where(weights > 0.0, real_x, jnp.inf), axis=-2)
        maximum = jnp.max(jnp.where(weights > 0.0, real_x, -jnp.inf), axis=-2)
        minimum = jnp.where(mass > 0.0, minimum, jnp.zeros_like(minimum))
        maximum = jnp.where(mass > 0.0, maximum, jnp.zeros_like(maximum))

        def configured(values, fallback, name):
            if values is None:
                return fallback
            if len(values) not in (1, batch.feature_count):
                raise ValueError(f"{name} must be scalar or match the feature count.")
            vector = jnp.asarray(values, dtype=real_x.dtype)
            if len(values) == 1:
                vector = jnp.broadcast_to(vector, (batch.feature_count,))
            return jnp.broadcast_to(vector, batch.case_shape + (batch.feature_count,))

        origin = configured(self.origin, minimum, "origin")
        raw_period = configured(self.period, maximum - minimum, "period")
        constant = raw_period == 0.0
        period = jnp.where(constant, jnp.ones_like(raw_period), raw_period)
        names = []
        if self.include_original:
            names.extend(batch.feature_schema.names)
        if self.include_bias:
            names.append("fourier_bias")
        names.extend(
            f"{name}_sin_{frequency}"
            for name in batch.feature_schema.names
            for frequency in range(1, self.n_frequencies + 1)
        )
        names.extend(
            f"{name}_cos_{frequency}"
            for name in batch.feature_schema.names
            for frequency in range(1, self.n_frequencies + 1)
        )
        output_schema = FeatureSchema(
            tuple(names),
            kinds=("continuous",) * len(names),
            layout_id=batch.feature_schema.layout_id,
        )
        model = FittedFourierFeatures(
            origin,
            period,
            n_frequencies=self.n_frequencies,
            include_bias=self.include_bias,
            include_original=self.include_original,
            input_schema=batch.feature_schema,
            output_schema=output_schema,
            case_shape=batch.case_shape,
        )
        diagnostics = _diagnostics(
            batch,
            output_schema,
            mass,
            effective,
            valid,
            status,
            method="fourier_features",
            constant=constant,
            details=(("n_frequencies", self.n_frequencies),),
        )
        fitted_range = self.period is None or self.origin is None
        return _fit_result(
            model,
            diagnostics,
            GradientContract(
                prediction_inputs="smooth",
                prediction_parameters="smooth",
                fit_features="almost-everywhere" if fitted_range else "none",
                fit_targets="none",
                fit_weights="none",
                fit_hyperparameters="none",
                fit_mode="direct",
                conditions=(
                    ("Extremum identities and positive-weight support are held fixed.",)
                    if fitted_range
                    else ()
                ),
            ),
        )


class FittedRandomFourierFeatures(AbstractArrayModel):
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)
    frequencies: Array
    phases: Array
    input_schema: FeatureSchema = eqx.field(static=True)
    output_schema: FeatureSchema = eqx.field(static=True)

    def __init__(
        self,
        frequencies: Array,
        phases: Array,
        /,
        *,
        input_schema: FeatureSchema,
        output_schema: FeatureSchema,
    ):
        self.in_size = len(input_schema.names)
        self.out_size = len(output_schema.names)
        self.frequencies = jnp.asarray(frequencies)
        self.phases = jnp.asarray(phases)
        self.input_schema = input_schema
        self.output_schema = output_schema

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        values = _check_features(x, self.in_size)
        return jnp.sqrt(2.0 / self.out_size) * jnp.cos(
            values @ self.frequencies + self.phases
        )

    def transform(self, x: Any, /, *, key: Any = None) -> Array:
        return self(x, key=key)

    def inverse_transform(self, x: Any, /, *, key: Any = None) -> Array:
        del x, key
        raise NotImplementedError(
            "Random Fourier features are periodic and not invertible."
        )


class RandomFourierFeatures(AbstractRecipe):
    """Gaussian-kernel random Fourier map; fitting requires an explicit JAX key."""

    n_components: int = eqx.field(static=True)
    gamma: Array
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        n_components: int = 100,
        *,
        gamma: ArrayLike = 1.0,
        weight_policy: WeightPolicy = "statistical",
    ):
        if int(n_components) <= 0:
            raise ValueError("n_components must be positive.")
        gamma_ = jnp.asarray(gamma, dtype=float)
        if gamma_.ndim != 0:
            raise ValueError("gamma must be scalar.")
        gamma_ = eqx.error_if(
            gamma_,
            ~jnp.isfinite(gamma_) | (gamma_ <= 0.0),
            "gamma must be finite and positive.",
        )
        if weight_policy not in ("none", "statistical", "measure", "product"):
            raise ValueError("Unsupported weight policy.")
        self.n_components = int(n_components)
        self.gamma = gamma_
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        if key is None:
            raise ValueError(
                "RandomFourierFeatures.fit_batch requires an explicit JAX key."
            )
        x, _weights, mass, effective, valid, status = _feature_observations(
            batch, weight_policy=self.weight_policy
        )
        frequency_key, phase_key = jax.random.split(key)
        dtype = x.real.dtype if jnp.issubdtype(x.dtype, jnp.inexact) else jnp.dtype(float)
        frequencies = jax.random.normal(
            frequency_key, (batch.feature_count, self.n_components), dtype=dtype
        ) * jnp.sqrt(2.0 * self.gamma)
        phases = jax.random.uniform(
            phase_key, (self.n_components,), dtype=dtype, minval=0.0, maxval=2.0 * jnp.pi
        )
        output_schema = FeatureSchema(
            tuple(f"random_fourier_{index}" for index in range(self.n_components)),
            kinds=("continuous",) * self.n_components,
            layout_id=batch.feature_schema.layout_id,
        )
        model = FittedRandomFourierFeatures(
            frequencies,
            phases,
            input_schema=batch.feature_schema,
            output_schema=output_schema,
        )
        diagnostics = _diagnostics(
            batch,
            output_schema,
            mass,
            effective,
            valid,
            status,
            method="random_fourier_features",
            details=(("n_components", self.n_components),),
        )
        return _fit_result(
            model,
            diagnostics,
            GradientContract(
                prediction_inputs="smooth",
                prediction_parameters="smooth",
                fit_features="none",
                fit_targets="none",
                fit_weights="none",
                fit_hyperparameters="smooth",
                fit_mode="direct",
                nondifferentiable_outputs=("random_frequencies", "random_phases"),
                conditions=("The explicit random key is held fixed.",),
            ),
        )


class FittedFeatureHasher(AbstractArrayModel, NonTrainableState):
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)
    buckets: Array
    signs: Array
    input_schema: FeatureSchema = eqx.field(static=True)
    output_schema: FeatureSchema = eqx.field(static=True)

    def __init__(
        self,
        buckets: Array,
        signs: Array,
        /,
        *,
        input_schema: FeatureSchema,
        output_schema: FeatureSchema,
    ):
        self.in_size = len(input_schema.names)
        self.out_size = len(output_schema.names)
        self.buckets = jnp.asarray(buckets, dtype=jnp.int32)
        self.signs = jnp.asarray(signs)
        self.input_schema = input_schema
        self.output_schema = output_schema

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        values = _check_features(x, self.in_size)
        output = jnp.zeros(
            values.shape[:-1] + (self.out_size,),
            dtype=jnp.result_type(values, self.signs),
        )
        return output.at[..., self.buckets].add(values * self.signs)

    def transform(self, x: Any, /, *, key: Any = None) -> Array:
        return self(x, key=key)

    def inverse_transform(self, x: Any, /, *, key: Any = None) -> Array:
        del x, key
        raise NotImplementedError("Feature hashing can collide and is not invertible.")


class FeatureHasher(AbstractRecipe):
    """Deterministic signed hashing of a fixed named feature axis."""

    n_features: int = eqx.field(static=True)
    alternate_sign: bool = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        n_features: int = 256,
        *,
        alternate_sign: bool = True,
        weight_policy: WeightPolicy = "statistical",
    ):
        if int(n_features) <= 0:
            raise ValueError("n_features must be positive.")
        if weight_policy not in ("none", "statistical", "measure", "product"):
            raise ValueError("Unsupported weight policy.")
        self.n_features = int(n_features)
        self.alternate_sign = bool(alternate_sign)
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        _x, _weights, mass, effective, valid, status = _feature_observations(
            batch, weight_policy=self.weight_policy
        )
        hashes = [
            int.from_bytes(
                blake2b(name.encode("utf-8"), digest_size=8).digest(), "little"
            )
            for name in batch.feature_schema.names
        ]
        buckets = jnp.asarray(
            [value % self.n_features for value in hashes], dtype=jnp.int32
        )
        sign_dtype = (
            _x.real.dtype if jnp.issubdtype(_x.dtype, jnp.inexact) else jnp.dtype(float)
        )
        signs = jnp.asarray(
            [
                -1.0 if self.alternate_sign and ((value >> 63) & 1) else 1.0
                for value in hashes
            ],
            dtype=sign_dtype,
        )
        output_schema = FeatureSchema(
            tuple(f"hash_{index}" for index in range(self.n_features)),
            kinds=("continuous",) * self.n_features,
            layout_id=batch.feature_schema.layout_id,
        )
        model = FittedFeatureHasher(
            buckets, signs, input_schema=batch.feature_schema, output_schema=output_schema
        )
        diagnostics = _diagnostics(
            batch,
            output_schema,
            mass,
            effective,
            valid,
            status,
            method="feature_hasher",
            details=(
                ("n_features", self.n_features),
                ("alternate_sign", self.alternate_sign),
            ),
        )
        return _fit_result(
            model,
            diagnostics,
            GradientContract(
                prediction_inputs="smooth",
                prediction_parameters="none",
                fit_features="none",
                fit_targets="none",
                fit_weights="none",
                fit_hyperparameters="none",
                fit_mode="direct",
                nondifferentiable_outputs=("hash_routes", "hash_signs"),
            ),
        )


class _AbstractRandomProjection(AbstractArrayModel):
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)
    projection: Array
    input_schema: FeatureSchema = eqx.field(static=True)
    output_schema: FeatureSchema = eqx.field(static=True)

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        return _check_features(x, self.in_size) @ self.projection

    def transform(self, x: Any, /, *, key: Any = None) -> Array:
        return self(x, key=key)

    def inverse_transform(self, x: Any, /, *, key: Any = None) -> Array:
        del x, key
        raise NotImplementedError(
            "Dimension-reducing random projection is not invertible."
        )


class FittedGaussianRandomProjection(_AbstractRandomProjection):
    def __init__(
        self,
        projection: Array,
        /,
        *,
        input_schema: FeatureSchema,
        output_schema: FeatureSchema,
    ):
        self.in_size = len(input_schema.names)
        self.out_size = len(output_schema.names)
        self.projection = jnp.asarray(projection)
        self.input_schema = input_schema
        self.output_schema = output_schema


class GaussianRandomProjection(AbstractRecipe):
    n_components: int = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(self, n_components: int, *, weight_policy: WeightPolicy = "statistical"):
        if int(n_components) <= 0:
            raise ValueError("n_components must be positive.")
        if weight_policy not in ("none", "statistical", "measure", "product"):
            raise ValueError("Unsupported weight policy.")
        self.n_components = int(n_components)
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        if key is None:
            raise ValueError(
                "GaussianRandomProjection.fit_batch requires an explicit JAX key."
            )
        x, _weights, mass, effective, valid, status = _feature_observations(
            batch, weight_policy=self.weight_policy
        )
        dtype = x.real.dtype if jnp.issubdtype(x.dtype, jnp.inexact) else jnp.dtype(float)
        projection = jax.random.normal(
            key, (batch.feature_count, self.n_components), dtype=dtype
        ) / jnp.sqrt(float(self.n_components))
        output_schema = FeatureSchema(
            tuple(f"gaussian_projection_{index}" for index in range(self.n_components)),
            kinds=("continuous",) * self.n_components,
            layout_id=batch.feature_schema.layout_id,
        )
        model = FittedGaussianRandomProjection(
            projection, input_schema=batch.feature_schema, output_schema=output_schema
        )
        diagnostics = _diagnostics(
            batch,
            output_schema,
            mass,
            effective,
            valid,
            status,
            method="gaussian_random_projection",
            details=(("n_components", self.n_components),),
        )
        return _fit_result(
            model,
            diagnostics,
            GradientContract(
                prediction_inputs="smooth",
                prediction_parameters="smooth",
                fit_features="none",
                fit_targets="none",
                fit_weights="none",
                fit_hyperparameters="none",
                fit_mode="direct",
                nondifferentiable_outputs=("gaussian_projection_draw",),
                conditions=("The explicit random key is held fixed.",),
            ),
        )


class FittedSparseRandomProjection(AbstractArrayModel):
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)
    projection: SparseLinearMap
    density: float = eqx.field(static=True)
    input_schema: FeatureSchema = eqx.field(static=True)
    output_schema: FeatureSchema = eqx.field(static=True)

    def __init__(
        self,
        projection: SparseLinearMap,
        /,
        *,
        density: float,
        input_schema: FeatureSchema,
        output_schema: FeatureSchema,
    ):
        self.in_size = len(input_schema.names)
        self.out_size = len(output_schema.names)
        self.projection = projection
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.density = float(density)

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        values = _check_features(x, self.in_size)
        source_first = jnp.moveaxis(values, -1, 0)
        return jnp.moveaxis(self.projection.mv(source_first), 0, -1)

    def transform(self, x: Any, /, *, key: Any = None) -> Array:
        return self(x, key=key)

    def inverse_transform(self, x: Any, /, *, key: Any = None) -> Array:
        del x, key
        raise NotImplementedError(
            "Dimension-reducing sparse random projection is not invertible."
        )


class SparseRandomProjection(AbstractRecipe):
    n_components: int = eqx.field(static=True)
    density: float | None = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        n_components: int,
        *,
        density: float | None = None,
        weight_policy: WeightPolicy = "statistical",
    ):
        if int(n_components) <= 0:
            raise ValueError("n_components must be positive.")
        if density is not None and (
            not jnp.isfinite(density) or not 0.0 < float(density) <= 1.0
        ):
            raise ValueError("density must lie in (0, 1].")
        if weight_policy not in ("none", "statistical", "measure", "product"):
            raise ValueError("Unsupported weight policy.")
        self.n_components = int(n_components)
        self.density = None if density is None else float(density)
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        if key is None:
            raise ValueError(
                "SparseRandomProjection.fit_batch requires an explicit JAX key."
            )
        x, _weights, mass, effective, valid, status = _feature_observations(
            batch, weight_policy=self.weight_policy
        )
        density = (
            min(1.0, 1.0 / (float(batch.feature_count) ** 0.5))
            if self.density is None
            else self.density
        )
        support_key, sign_key = jax.random.split(key)
        support = jax.random.bernoulli(
            support_key, density, (batch.feature_count, self.n_components)
        )
        signs = jax.random.bernoulli(
            sign_key, 0.5, (batch.feature_count, self.n_components)
        )
        dtype = x.real.dtype if jnp.issubdtype(x.dtype, jnp.inexact) else jnp.dtype(float)
        coefficients = jnp.where(signs, 1.0, -1.0).astype(dtype).reshape((-1,))
        coefficients = coefficients / jnp.sqrt(float(self.n_components) * density)
        relation = EdgeRelation(
            jnp.repeat(
                jnp.arange(batch.feature_count, dtype=jnp.int32), self.n_components
            ),
            jnp.tile(jnp.arange(self.n_components, dtype=jnp.int32), batch.feature_count),
            source_size=batch.feature_count,
            target_size=self.n_components,
            valid=support.reshape((-1,)),
        )
        projection = SparseLinearMap(relation, coefficients)
        output_schema = FeatureSchema(
            tuple(f"sparse_projection_{index}" for index in range(self.n_components)),
            kinds=("continuous",) * self.n_components,
            layout_id=batch.feature_schema.layout_id,
        )
        model = FittedSparseRandomProjection(
            projection,
            density=float(density),
            input_schema=batch.feature_schema,
            output_schema=output_schema,
        )
        diagnostics = _diagnostics(
            batch,
            output_schema,
            mass,
            effective,
            valid,
            status,
            method="sparse_random_projection",
            details=(("n_components", self.n_components), ("density", float(density))),
        )
        return _fit_result(
            model,
            diagnostics,
            GradientContract(
                prediction_inputs="smooth",
                prediction_parameters="smooth",
                fit_features="none",
                fit_targets="none",
                fit_weights="none",
                fit_hyperparameters="none",
                fit_mode="direct",
                nondifferentiable_outputs=(
                    "sparse_projection_support",
                    "sparse_projection_signs",
                ),
                conditions=("The explicit random key is held fixed.",),
            ),
        )


def _box_cox(values: Array, lambdas: Array) -> Array:
    log_values = jnp.log(values)
    safe_lambda = jnp.where(jnp.abs(lambdas) > 1e-7, lambdas, 1.0)
    powered = jnp.expm1(lambdas * log_values) / safe_lambda
    return jnp.where(jnp.abs(lambdas) > 1e-7, powered, log_values)


def _box_cox_inverse(values: Array, lambdas: Array) -> Array:
    argument = 1.0 + lambdas * values
    safe_lambda = jnp.where(jnp.abs(lambdas) > 1e-7, lambdas, 1.0)
    powered = jnp.exp(
        jnp.log(jnp.maximum(argument, jnp.finfo(values.real.dtype).tiny)) / safe_lambda
    )
    return jnp.where(jnp.abs(lambdas) > 1e-7, powered, jnp.exp(values))


def _yeo_johnson(values: Array, lambdas: Array) -> Array:
    positive_log = jnp.log1p(jnp.maximum(values, 0.0))
    negative_log = jnp.log1p(jnp.maximum(-values, 0.0))
    safe_lambda = jnp.where(jnp.abs(lambdas) > 1e-7, lambdas, 1.0)
    positive_power = jnp.expm1(lambdas * positive_log) / safe_lambda
    positive = jnp.where(jnp.abs(lambdas) > 1e-7, positive_power, positive_log)
    complement = 2.0 - lambdas
    safe_complement = jnp.where(jnp.abs(complement) > 1e-7, complement, 1.0)
    negative_power = -jnp.expm1(complement * negative_log) / safe_complement
    negative = jnp.where(jnp.abs(complement) > 1e-7, negative_power, -negative_log)
    return jnp.where(values >= 0.0, positive, negative)


def _yeo_johnson_inverse(values: Array, lambdas: Array) -> Array:
    positive_argument = 1.0 + lambdas * values
    safe_lambda = jnp.where(jnp.abs(lambdas) > 1e-7, lambdas, 1.0)
    positive_power = jnp.expm1(
        jnp.log(jnp.maximum(positive_argument, jnp.finfo(values.dtype).tiny))
        / safe_lambda
    )
    positive = jnp.where(jnp.abs(lambdas) > 1e-7, positive_power, jnp.expm1(values))
    complement = 2.0 - lambdas
    negative_argument = 1.0 - complement * values
    safe_complement = jnp.where(jnp.abs(complement) > 1e-7, complement, 1.0)
    negative_power = -jnp.expm1(
        jnp.log(jnp.maximum(negative_argument, jnp.finfo(values.dtype).tiny))
        / safe_complement
    )
    negative = jnp.where(jnp.abs(complement) > 1e-7, negative_power, -jnp.expm1(-values))
    return jnp.where(values >= 0.0, positive, negative)


class FittedPowerTransformer(AbstractArrayModel):
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)
    lambdas: Array
    method: Literal["yeo-johnson", "box-cox"] = eqx.field(static=True)
    input_schema: FeatureSchema = eqx.field(static=True)
    output_schema: FeatureSchema = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        lambdas: Array,
        /,
        *,
        method: Literal["yeo-johnson", "box-cox"],
        schema: FeatureSchema,
        case_shape: tuple[int, ...],
    ):
        self.in_size = len(schema.names)
        self.out_size = len(schema.names)
        self.lambdas = jnp.asarray(lambdas)
        self.method = method
        self.input_schema = schema
        self.output_schema = schema
        self.case_shape = tuple(case_shape)

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        values = _check_features(x, self.in_size)
        if jnp.issubdtype(values.dtype, jnp.complexfloating):
            raise TypeError("PowerTransformer requires real-valued features.")
        values = values.astype(jnp.result_type(values, float))
        lambdas = _align_parameter(self.lambdas, values, self.case_shape)
        if self.method == "box-cox":
            values = eqx.error_if(
                values,
                jnp.any(values <= 0.0),
                "Box-Cox inputs must be strictly positive.",
            )
            return _box_cox(values, lambdas)
        return _yeo_johnson(values, lambdas)

    def transform(self, x: Any, /, *, key: Any = None) -> Array:
        return self(x, key=key)

    def inverse_transform(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        values = _check_features(x, self.out_size)
        values = values.astype(jnp.result_type(values, float))
        lambdas = _align_parameter(self.lambdas, values, self.case_shape)
        if self.method == "box-cox":
            domain = (jnp.abs(lambdas) <= 1e-7) | (1.0 + lambdas * values > 0.0)
            values = eqx.error_if(
                values, jnp.any(~domain), "Values lie outside the fitted Box-Cox range."
            )
            return _box_cox_inverse(values, lambdas)
        positive_domain = (
            (values < 0.0) | (jnp.abs(lambdas) <= 1e-7) | (1.0 + lambdas * values > 0.0)
        )
        complement = 2.0 - lambdas
        negative_domain = (
            (values >= 0.0)
            | (jnp.abs(complement) <= 1e-7)
            | (1.0 - complement * values > 0.0)
        )
        values = eqx.error_if(
            values,
            jnp.any(~positive_domain | ~negative_domain),
            "Values lie outside the fitted Yeo-Johnson range.",
        )
        return _yeo_johnson_inverse(values, lambdas)


class PowerTransformer(AbstractRecipe):
    """Hard fixed-grid weighted likelihood selection with differentiable power apply."""

    method: Literal["yeo-johnson", "box-cox"] = eqx.field(static=True)
    lambda_range: tuple[float, float] = eqx.field(static=True)
    n_lambdas: int = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        method: Literal["yeo-johnson", "box-cox"] = "yeo-johnson",
        *,
        lambda_range: tuple[float, float] = (-2.0, 2.0),
        n_lambdas: int = 65,
        weight_policy: WeightPolicy = "statistical",
    ):
        if method not in ("yeo-johnson", "box-cox"):
            raise ValueError("method must be 'yeo-johnson' or 'box-cox'.")
        lower, upper = float(lambda_range[0]), float(lambda_range[1])
        if not jnp.isfinite(lower) or not jnp.isfinite(upper) or not upper > lower:
            raise ValueError("lambda_range must be finite and increasing.")
        if int(n_lambdas) < 2:
            raise ValueError("n_lambdas must be at least two.")
        if weight_policy not in ("none", "statistical", "measure", "product"):
            raise ValueError("Unsupported weight policy.")
        self.method = method
        self.lambda_range = (lower, upper)
        self.n_lambdas = int(n_lambdas)
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        x, weights, mass, effective, valid, status = _feature_observations(
            batch, weight_policy=self.weight_policy
        )
        if jnp.issubdtype(x.dtype, jnp.complexfloating):
            raise TypeError("PowerTransformer requires real-valued features.")
        x = x.astype(jnp.result_type(x, float))
        positive = jnp.all((x > 0.0) | (weights == 0.0), axis=-2)
        if self.method == "box-cox":
            positive_case = jnp.all(positive, axis=-1)
            valid = valid & positive_case
            status = jnp.where(positive_case, status, ML_INFEASIBLE)
            safe_x = jnp.where(
                weights > 0.0, jnp.maximum(x, jnp.finfo(x.dtype).tiny), 1.0
            )
        else:
            safe_x = x
        lambdas = jnp.linspace(
            self.lambda_range[0],
            self.lambda_range[1],
            self.n_lambdas,
            dtype=jnp.result_type(x, float),
        )
        expanded = safe_x[..., None]
        lambda_bank = lambdas.reshape((1,) * x.ndim + (self.n_lambdas,))
        transformed = (
            _box_cox(expanded, lambda_bank)
            if self.method == "box-cox"
            else _yeo_johnson(expanded, lambda_bank)
        )
        expanded_weight = weights[..., None]
        mean = jnp.sum(expanded_weight * transformed, axis=-3) / jnp.maximum(
            mass[..., None], jnp.finfo(weights.dtype).tiny
        )
        residual = jnp.where(
            expanded_weight > 0.0, transformed - mean[..., None, :, :], 0.0
        )
        variance = jnp.sum(expanded_weight * residual * residual, axis=-3) / jnp.maximum(
            mass[..., None], jnp.finfo(weights.dtype).tiny
        )
        if self.method == "box-cox":
            log_base = jnp.log(safe_x)
            jacobian = jnp.sum(
                weights[..., None] * (lambdas - 1.0) * log_base[..., None], axis=-3
            )
        else:
            jacobian_log = jnp.where(
                safe_x >= 0.0, jnp.log1p(safe_x), -jnp.log1p(-safe_x)
            )
            jacobian = jnp.sum(
                weights[..., None] * (lambdas - 1.0) * jacobian_log[..., None], axis=-3
            )
        likelihood = (
            -0.5
            * mass[..., None]
            * jnp.log(jnp.maximum(variance, jnp.finfo(variance.dtype).tiny))
            + jacobian
        )
        selected = jnp.argmax(likelihood, axis=-1)
        fitted_lambdas = lambdas[selected]
        selected_variance = jnp.take_along_axis(variance, selected[..., None], axis=-1)[
            ..., 0
        ]
        constant = selected_variance <= jnp.finfo(selected_variance.dtype).eps
        model = FittedPowerTransformer(
            fitted_lambdas,
            method=self.method,
            schema=batch.feature_schema,
            case_shape=batch.case_shape,
        )
        diagnostics = _diagnostics(
            batch,
            batch.feature_schema,
            mass,
            effective,
            valid,
            status,
            method="power_transformer",
            constant=constant,
            details=(("power_method", self.method), ("n_lambdas", self.n_lambdas)),
        )
        return _fit_result(
            model,
            diagnostics,
            GradientContract(
                prediction_inputs="almost-everywhere",
                prediction_parameters="smooth",
                fit_features="none",
                fit_weights="none",
                fit_hyperparameters="none",
                fit_mode="stopped",
                nondifferentiable_outputs=("selected_lambda",),
                conditions=("The fixed-grid maximum-likelihood lambda is held fixed.",),
            ),
        )


class FittedQuantileTransformer(AbstractArrayModel):
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)
    quantiles: Array
    references: Array
    output_distribution: Literal["uniform", "normal"] = eqx.field(static=True)
    input_schema: FeatureSchema = eqx.field(static=True)
    output_schema: FeatureSchema = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        quantiles: Array,
        references: Array,
        /,
        *,
        output_distribution: Literal["uniform", "normal"],
        schema: FeatureSchema,
        case_shape: tuple[int, ...],
    ):
        self.in_size = len(schema.names)
        self.out_size = len(schema.names)
        self.quantiles = jnp.asarray(quantiles)
        self.references = jnp.asarray(references)
        self.output_distribution = output_distribution
        self.input_schema = schema
        self.output_schema = schema
        self.case_shape = tuple(case_shape)

    def _uniform(self, values: Array) -> Array:
        quantiles = _align_parameter(
            self.quantiles, values, self.case_shape, trailing_rank=2
        )
        quantiles = jnp.broadcast_to(
            quantiles, values.shape + (self.quantiles.shape[-1],)
        )
        flat_values = values.reshape((-1,))
        flat_quantiles = quantiles.reshape((-1, quantiles.shape[-1]))
        return jax.vmap(lambda value, knots: jnp.interp(value, knots, self.references))(
            flat_values, flat_quantiles
        ).reshape(values.shape)

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        values = _check_features(x, self.in_size)
        if jnp.issubdtype(values.dtype, jnp.complexfloating):
            raise TypeError("QuantileTransformer requires real-valued features.")
        uniform = self._uniform(values)
        if self.output_distribution == "normal":
            epsilon = jnp.finfo(uniform.dtype).eps
            return jsp.special.ndtri(jnp.clip(uniform, epsilon, 1.0 - epsilon))
        return uniform

    def transform(self, x: Any, /, *, key: Any = None) -> Array:
        return self(x, key=key)

    def inverse_transform(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        values = _check_features(x, self.out_size)
        quantiles = _align_parameter(
            self.quantiles, values, self.case_shape, trailing_rank=2
        )
        quantiles = jnp.broadcast_to(
            quantiles, values.shape + (self.quantiles.shape[-1],)
        )
        values = eqx.error_if(
            values,
            jnp.any(jnp.diff(quantiles, axis=-1) <= 0.0),
            "Fitted quantiles contain ties, so the quantile transform is not bijective.",
        )
        probability = (
            jsp.special.ndtr(values) if self.output_distribution == "normal" else values
        )
        probability = jnp.clip(probability, 0.0, 1.0)
        flat_probability = probability.reshape((-1,))
        flat_quantiles = quantiles.reshape((-1, quantiles.shape[-1]))
        return jax.vmap(lambda value, knots: jnp.interp(value, self.references, knots))(
            flat_probability, flat_quantiles
        ).reshape(values.shape)


class QuantileTransformer(AbstractRecipe):
    """Fixed-capacity weighted empirical CDF with uniform or Gaussian output."""

    n_quantiles: int = eqx.field(static=True)
    output_distribution: Literal["uniform", "normal"] = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        n_quantiles: int = 100,
        *,
        output_distribution: Literal["uniform", "normal"] = "uniform",
        weight_policy: WeightPolicy = "statistical",
    ):
        if int(n_quantiles) < 2:
            raise ValueError("n_quantiles must be at least two.")
        if output_distribution not in ("uniform", "normal"):
            raise ValueError("output_distribution must be 'uniform' or 'normal'.")
        if weight_policy not in ("none", "statistical", "measure", "product"):
            raise ValueError("Unsupported weight policy.")
        self.n_quantiles = int(n_quantiles)
        self.output_distribution = output_distribution
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        x, weights, mass, effective, valid, status = _feature_observations(
            batch, weight_policy=self.weight_policy
        )
        if jnp.issubdtype(x.dtype, jnp.complexfloating):
            raise TypeError("QuantileTransformer requires real-valued features.")
        references = jnp.linspace(0.0, 1.0, self.n_quantiles, dtype=weights.dtype)
        quantiles = _weighted_quantiles(x, weights, references)
        constant = jnp.any(jnp.diff(quantiles, axis=-1) <= 0.0, axis=-1)
        model = FittedQuantileTransformer(
            quantiles,
            references,
            output_distribution=self.output_distribution,
            schema=batch.feature_schema,
            case_shape=batch.case_shape,
        )
        diagnostics = _diagnostics(
            batch,
            batch.feature_schema,
            mass,
            effective,
            valid,
            status,
            method="quantile_transformer",
            constant=constant,
            details=(
                ("n_quantiles", self.n_quantiles),
                ("output_distribution", self.output_distribution),
            ),
        )
        return _fit_result(
            model,
            diagnostics,
            GradientContract(
                prediction_inputs="almost-everywhere",
                prediction_parameters="conditional",
                fit_features="none",
                fit_weights="none",
                fit_mode="stopped",
                nondifferentiable_outputs=("weighted_order_statistics",),
                conditions=(
                    "Empirical quantile order and interpolation intervals are held fixed.",
                ),
            ),
        )


__all__ = [
    "FeatureHasher",
    "FittedFeatureHasher",
    "FittedFourierFeatures",
    "FittedGaussianRandomProjection",
    "FittedPolynomialFeatures",
    "FittedPowerTransformer",
    "FittedQuantileTransformer",
    "FittedRandomFourierFeatures",
    "FittedSparseRandomProjection",
    "FittedSplineTransformer",
    "FourierFeatures",
    "GaussianRandomProjection",
    "PolynomialFeatures",
    "PowerTransformer",
    "QuantileTransformer",
    "RandomFourierFeatures",
    "SparseRandomProjection",
    "SplineTransformer",
]
