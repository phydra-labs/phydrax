#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import coordax as cx
import jax
import jax.numpy as jnp

from .._frozendict import frozendict
from .._strict import StrictModule
from ..domain._sampling import get_sampler
from ._distributions import AbstractDistribution
from ._predictive import PredictiveField, SampleAxis


class RandomSampleBatch(StrictModule):
    """Aligned scalar random-variable draws from one joint unit-cube design."""

    values: frozendict[str, Any]
    sample_dim: str
    distributions: frozendict[str, AbstractDistribution]

    def __init__(
        self,
        values: Mapping[str, Any],
        /,
        *,
        sample_dim: str,
        distributions: Mapping[str, AbstractDistribution],
    ):
        names = tuple(values)
        if not names:
            raise ValueError("RandomSampleBatch values must be non-empty.")
        if tuple(distributions) != names:
            raise ValueError(
                "values and distributions must have identical ordered labels."
            )
        if not isinstance(sample_dim, str) or not sample_dim:
            raise ValueError("sample_dim must be a non-empty string.")
        if sample_dim in names:
            raise ValueError("sample_dim must not collide with a random-variable label.")
        arrays = {name: jnp.asarray(values[name], dtype=float) for name in names}
        sizes = {int(array.shape[0]) for array in arrays.values() if array.ndim == 1}
        if any(array.ndim != 1 for array in arrays.values()) or len(sizes) != 1:
            raise ValueError("Every random-variable sample must be an aligned 1D array.")
        if next(iter(sizes)) <= 0:
            raise ValueError("RandomSampleBatch must contain at least one sample.")
        self.values = frozendict(arrays)
        self.sample_dim = sample_dim
        self.distributions = frozendict(distributions)

    @property
    def num_samples(self) -> int:
        return int(next(iter(self.values.values())).shape[0])


def sample_joint(
    distributions: Mapping[str, AbstractDistribution],
    /,
    *,
    num_samples: int,
    key,
    sampler: str = "sobol_scrambled",
    sample_dim: str = "__phydra_uq_input",
) -> RandomSampleBatch:
    """Transform one ``d``-dimensional low-discrepancy design through marginal ICDFs."""
    names = tuple(distributions)
    if not names:
        raise ValueError("distributions must be non-empty.")
    count = int(num_samples)
    if count <= 0:
        raise ValueError("num_samples must be positive.")
    for name, distribution in distributions.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Distribution labels must be non-empty strings.")
        if not isinstance(distribution, AbstractDistribution):
            raise TypeError(f"Distribution {name!r} must implement AbstractDistribution.")
    unit_design = get_sampler(sampler)(count, len(names), key)
    values = {
        name: distribution.icdf(unit_design[:, index])
        for index, (name, distribution) in enumerate(distributions.items())
    }
    return RandomSampleBatch(
        values,
        sample_dim=sample_dim,
        distributions=distributions,
    )


def propagate(
    function,
    samples: RandomSampleBatch,
    /,
    *,
    batch_size: int | None = None,
    valid_policy: Literal["record", "raise"] = "record",
    call_style: Literal["keywords", "mapping"] = "keywords",
    **kwargs: Any,
) -> PredictiveField:
    """Propagate aligned uncertain inputs into coherent output realizations."""
    if not callable(function):
        raise TypeError("function must be callable.")
    if not isinstance(samples, RandomSampleBatch):
        raise TypeError("samples must be a RandomSampleBatch.")
    if valid_policy not in ("record", "raise"):
        raise ValueError("valid_policy must be 'record' or 'raise'.")
    if call_style not in ("keywords", "mapping"):
        raise ValueError("call_style must be 'keywords' or 'mapping'.")
    count = samples.num_samples
    chunk = count if batch_size is None else int(batch_size)
    if chunk <= 0:
        raise ValueError("batch_size must be positive.")
    names = tuple(samples.values)
    data_parts = []

    def evaluate(*values):
        arguments = {name: value for name, value in zip(names, values, strict=True)}
        if call_style == "keywords":
            return function(**arguments, **kwargs)
        return function(frozendict(arguments), **kwargs)

    first_values = tuple(samples.values[name][0] for name in names)
    template = evaluate(*first_values)
    if isinstance(template, cx.Field):
        template_data = jnp.asarray(template.data)
        template_dims = tuple(template.dims)
        returns_field = True
    else:
        try:
            template_data = jnp.asarray(template)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "propagate function must return one array or coordax.Field."
            ) from exc
        template_dims = (None,) * template_data.ndim
        returns_field = False

    def evaluate_data(*values):
        result = evaluate(*values)
        if returns_field:
            if not isinstance(result, cx.Field):
                raise TypeError("Propagated output type changed between samples.")
            if result.dims != template_dims:
                raise ValueError("Propagated field dimensions changed between samples.")
            return result.data
        return jnp.asarray(result)

    for start in range(0, count, chunk):
        stop = min(start + chunk, count)
        columns = tuple(samples.values[name][start:stop] for name in names)
        data = jnp.asarray(jax.vmap(evaluate_data)(*columns))
        if int(data.shape[0]) != stop - start:
            raise ValueError("Propagated output did not retain the leading sample axis.")
        if data.shape[1:] != template_data.shape:
            raise ValueError("Propagated chunks produced inconsistent output structure.")
        data_parts.append(data)

    data = jnp.concatenate(tuple(data_parts), axis=0)
    if template_dims is None:
        raise RuntimeError("Propagation produced no output chunks.")
    valid_data = jnp.all(jnp.isfinite(data).reshape((count, -1)), axis=1)
    if valid_policy == "raise" and not bool(jnp.all(valid_data)):
        failed = tuple(int(index) for index in jnp.where(~valid_data)[0])
        raise FloatingPointError(f"Propagation produced invalid samples at {failed!r}.")
    sample_field = cx.Field(data, dims=(samples.sample_dim, *template_dims))
    valid = cx.Field(valid_data, dims=(samples.sample_dim,))
    return PredictiveField(
        sample_field,
        (SampleAxis(samples.sample_dim, "input"),),
        valid=valid,
    )


__all__ = ["RandomSampleBatch", "propagate", "sample_joint"]
