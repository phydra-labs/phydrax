#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import coordax as cx
import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike

from .._frozendict import frozendict
from .._strict import StrictModule
from ..domain._sampling import get_sampler
from ._distributions import AbstractDistribution


class SobolResult(StrictModule):
    """First-order and total-order global sensitivity fields."""

    parameter_names: tuple[str, ...]
    first_order: cx.Field
    total_order: cx.Field
    output_variance: cx.Field
    num_samples: int
    parameter_dim: str

    def __init__(
        self,
        *,
        parameter_names: tuple[str, ...],
        first_order: cx.Field,
        total_order: cx.Field,
        output_variance: cx.Field,
        num_samples: int,
        parameter_dim: str,
    ):
        expected = len(parameter_names)
        if (
            first_order.dims != total_order.dims
            or first_order.data.shape != total_order.data.shape
        ):
            raise ValueError("Sobol first-order and total-order fields must align.")
        if (
            first_order.dims[0] != parameter_dim
            or int(first_order.data.shape[0]) != expected
        ):
            raise ValueError(
                "Sobol result parameter axis does not match parameter_names."
            )
        if first_order.dims[1:] != output_variance.dims:
            raise ValueError("Sobol output dimensions do not match output variance.")
        self.parameter_names = parameter_names
        self.first_order = first_order
        self.total_order = total_order
        self.output_variance = output_variance
        self.num_samples = int(num_samples)
        self.parameter_dim = parameter_dim


def sobol_indices(
    function,
    distributions: Mapping[str, AbstractDistribution],
    /,
    *,
    num_samples: int,
    key,
    sampler: str = "sobol_scrambled",
    batch_size: int | None = None,
    parameter_dim: str = "__phydra_uq_parameter",
    call_style: Literal["keywords", "mapping"] = "keywords",
    reduce_output: Literal["mean", "sum"] | None = None,
    mask: ArrayLike | None = None,
    weights: ArrayLike | None = None,
    **kwargs: Any,
) -> SobolResult:
    """Saltelli first-order and Jansen total-order indices from one joint QMC design."""
    if not callable(function):
        raise TypeError("function must be callable.")
    names = tuple(distributions)
    if not names:
        raise ValueError("distributions must be non-empty.")
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("Distribution labels must be non-empty strings.")
    if not isinstance(parameter_dim, str):
        raise TypeError("parameter_dim must be a string.")
    if parameter_dim in names or not parameter_dim:
        raise ValueError(
            "parameter_dim must be non-empty and distinct from input labels."
        )
    count = int(num_samples)
    if count < 2:
        raise ValueError("num_samples must be at least two.")
    if reduce_output not in (None, "mean", "sum"):
        raise ValueError("reduce_output must be None, 'mean', or 'sum'.")
    if call_style not in ("keywords", "mapping"):
        raise ValueError("call_style must be 'keywords' or 'mapping'.")
    for name, distribution in distributions.items():
        if not isinstance(distribution, AbstractDistribution):
            raise TypeError(f"Distribution {name!r} must implement AbstractDistribution.")
    dimension = len(names)
    unit = get_sampler(sampler)(count, 2 * dimension, key)
    a = jnp.stack(
        tuple(
            distribution.icdf(unit[:, index])
            for index, distribution in enumerate(distributions.values())
        ),
        axis=1,
    )
    b = jnp.stack(
        tuple(
            distribution.icdf(unit[:, dimension + index])
            for index, distribution in enumerate(distributions.values())
        ),
        axis=1,
    )
    f_a, output_dims = _evaluate_design(
        function,
        names,
        a,
        batch_size=batch_size,
        call_style=call_style,
        **kwargs,
    )
    f_b, b_dims = _evaluate_design(
        function,
        names,
        b,
        batch_size=batch_size,
        call_style=call_style,
        **kwargs,
    )
    if f_a.shape != f_b.shape or output_dims != b_dims:
        raise ValueError("A and B designs produced inconsistent output structure.")
    if reduce_output is not None:
        f_a = _reduce_sample_outputs(
            f_a, reduction=reduce_output, mask=mask, weights=weights
        )
        f_b = _reduce_sample_outputs(
            f_b, reduction=reduce_output, mask=mask, weights=weights
        )
        output_dims = ()
    elif mask is not None or weights is not None:
        raise ValueError("mask and weights require reduce_output='mean' or 'sum'.")
    combined = jnp.concatenate((f_a, f_b), axis=0)
    variance = jnp.var(combined, axis=0, ddof=1)
    tolerance = jnp.finfo(variance.dtype).eps * jnp.maximum(
        1.0, jnp.mean(combined**2, axis=0)
    )
    if bool(jnp.any(~jnp.isfinite(variance))) or bool(jnp.any(variance <= tolerance)):
        raise ValueError("Sobol indices require finite, non-zero output variance.")
    first = []
    total = []
    for index in range(dimension):
        hybrid = a.at[:, index].set(b[:, index])
        f_ab, hybrid_dims = _evaluate_design(
            function,
            names,
            hybrid,
            batch_size=batch_size,
            call_style=call_style,
            **kwargs,
        )
        if reduce_output is not None:
            f_ab = _reduce_sample_outputs(
                f_ab, reduction=reduce_output, mask=mask, weights=weights
            )
            hybrid_dims = ()
        if f_ab.shape != f_a.shape or hybrid_dims != output_dims:
            raise ValueError(
                "Hybrid Sobol design produced inconsistent output structure."
            )
        first.append(jnp.mean(f_b * (f_ab - f_a), axis=0) / variance)
        total.append(0.5 * jnp.mean((f_a - f_ab) ** 2, axis=0) / variance)
    first_data = jnp.stack(tuple(first), axis=0)
    total_data = jnp.stack(tuple(total), axis=0)
    return SobolResult(
        parameter_names=names,
        first_order=cx.Field(first_data, dims=(parameter_dim, *output_dims)),
        total_order=cx.Field(total_data, dims=(parameter_dim, *output_dims)),
        output_variance=cx.Field(variance, dims=output_dims),
        num_samples=count,
        parameter_dim=parameter_dim,
    )


def _evaluate_design(
    function,
    names: tuple[str, ...],
    design,
    /,
    *,
    batch_size: int | None,
    call_style: Literal["keywords", "mapping"],
    **kwargs: Any,
):
    count = int(design.shape[0])
    chunk = count if batch_size is None else int(batch_size)
    if chunk <= 0:
        raise ValueError("batch_size must be positive.")

    def evaluate(row):
        arguments = {name: row[index] for index, name in enumerate(names)}
        if call_style == "keywords":
            return function(**arguments, **kwargs)
        return function(frozendict(arguments), **kwargs)

    template = evaluate(design[0])
    if isinstance(template, cx.Field):
        template_data = jnp.asarray(template.data)
        output_dims = tuple(template.dims)
        returns_field = True
    else:
        try:
            template_data = jnp.asarray(template)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "Sensitivity function must return one array or coordax.Field."
            ) from exc
        output_dims = (None,) * template_data.ndim
        returns_field = False

    def evaluate_data(row):
        value = evaluate(row)
        if returns_field:
            if not isinstance(value, cx.Field):
                raise TypeError("Sensitivity output type changed between samples.")
            if value.dims != output_dims:
                raise ValueError("Sensitivity field dimensions changed between samples.")
            return value.data
        return jnp.asarray(value)

    parts = []
    for start in range(0, count, chunk):
        data = jnp.asarray(jax.vmap(evaluate_data)(design[start : start + chunk]))
        if data.shape[1:] != template_data.shape:
            raise ValueError("Sensitivity chunks produced inconsistent output shape.")
        parts.append(data)
    values = jnp.concatenate(tuple(parts), axis=0)
    if bool(jnp.any(~jnp.isfinite(values))):
        raise FloatingPointError("Sensitivity evaluation produced non-finite outputs.")
    return values, output_dims


def _reduce_sample_outputs(
    values,
    /,
    *,
    reduction: Literal["mean", "sum"],
    mask: ArrayLike | None,
    weights: ArrayLike | None,
):
    output_shape = values.shape[1:]
    effective = jnp.ones(output_shape, dtype=values.dtype)
    if mask is not None:
        effective = effective * jnp.broadcast_to(
            jnp.asarray(mask, dtype=bool), output_shape
        )
    if weights is not None:
        weight_array = jnp.broadcast_to(jnp.asarray(weights, dtype=float), output_shape)
        if bool(jnp.any(~jnp.isfinite(weight_array))) or bool(
            jnp.any(weight_array < 0.0)
        ):
            raise ValueError("weights must be finite and non-negative.")
        effective = effective * weight_array
    flat_values = values.reshape((int(values.shape[0]), -1))
    flat_weight = effective.reshape((-1,))
    sums = jnp.sum(flat_values * flat_weight, axis=1)
    if reduction == "sum":
        return sums
    denominator = jnp.sum(flat_weight)
    if not bool(denominator > 0.0):
        raise ValueError("Sensitivity reduction has zero total weight.")
    return sums / denominator


__all__ = ["SobolResult", "sobol_indices"]
