#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import coordax as cx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PyTree

from .._frozendict import frozendict
from ._posterior import PosteriorProblem
from ._predictive import PredictiveField, SampleAxis, UncertaintySource


def predict_from_position_samples(
    problem: PosteriorProblem,
    positions: PyTree[Array],
    /,
    *args: Any,
    sample_dims: Sequence[str],
    sample_sources: Sequence[UncertaintySource],
    batch_size: int | None = None,
    valid_policy: Literal["record", "raise"] = "record",
    **kwargs: Any,
) -> PredictiveField | frozendict[str, PredictiveField]:
    """Evaluate latent predictions and any declared conditional measurement variance."""
    dimensions, sources = _sample_metadata(sample_dims, sample_sources)
    predictions = _evaluate_position_callback(
        positions,
        lambda position: problem.predict(position, *args, **kwargs),
        sample_dims=dimensions,
        sample_sources=sources,
        batch_size=batch_size,
        valid_policy=valid_policy,
        owner="Posterior prediction",
    )
    if problem.observation_variance_fn is None:
        return predictions
    conditional = _evaluate_position_callback(
        positions,
        lambda position: problem.conditional_observation_variance(
            position, *args, **kwargs
        ),
        sample_dims=dimensions,
        sample_sources=sources,
        batch_size=batch_size,
        valid_policy=valid_policy,
        owner="Conditional observation variance",
    )
    return _attach_conditional_variance(predictions, conditional)


def sample_observations_from_position_samples(
    problem: PosteriorProblem,
    key: Array,
    positions: PyTree[Array],
    /,
    *args: Any,
    num_observation_samples: int,
    sample_dims: Sequence[str],
    sample_sources: Sequence[UncertaintySource],
    observation_dim: str = "__phydra_uq_observation",
    batch_size: int | None = None,
    valid_policy: Literal["record", "raise"] = "record",
    **kwargs: Any,
) -> PredictiveField | frozendict[str, PredictiveField]:
    """Draw measurement noise while preserving posterior and observation axes."""
    if problem.sample_observation_fn is None:
        raise ValueError("PosteriorProblem has no observation-sampling function.")
    dimensions, sources = _sample_metadata(sample_dims, sample_sources)
    count = int(num_observation_samples)
    if count <= 0:
        raise ValueError("num_observation_samples must be positive.")
    observation_dimension = str(observation_dim)
    if not observation_dimension or observation_dimension in dimensions:
        raise ValueError(
            "observation_dim must be non-empty and distinct from posterior sample dims."
        )
    first_shape, total, flat_positions = _position_sample_layout(positions, dimensions)
    chunk = _batch_size(batch_size, total)
    split_keys = jr.split(key, total * count)
    keys = split_keys.reshape((total, count, *split_keys.shape[1:]))
    first_position = jax.tree_util.tree_map(lambda value: value[0], flat_positions)
    templates, single = _prediction_mapping(
        problem.sample_observation(keys[0, 0], first_position, *args, **kwargs)
    )
    names = tuple(templates)
    parts: dict[str, list[Array]] = {name: [] for name in names}

    def evaluate(position, draw_keys):
        def draw(draw_key):
            result, result_single = _prediction_mapping(
                problem.sample_observation(
                    draw_key,
                    position,
                    *args,
                    **kwargs,
                )
            )
            if result_single != single or tuple(result) != names:
                raise ValueError("Posterior observation structure changed between draws.")
            return tuple(jnp.asarray(result[name].data) for name in names)

        return jax.vmap(draw)(draw_keys)

    for start in range(0, total, chunk):
        stop = min(start + chunk, total)
        selected = jax.tree_util.tree_map(lambda value: value[start:stop], flat_positions)
        evaluated = jax.vmap(evaluate)(selected, keys[start:stop])
        for name, data in zip(names, evaluated, strict=True):
            parts[name].append(jnp.asarray(data))

    predictions: dict[str, PredictiveField] = {}
    all_dimensions = (*dimensions, observation_dimension)
    sample_axes = (
        *(
            SampleAxis(dim, source)
            for dim, source in zip(dimensions, sources, strict=True)
        ),
        SampleAxis(observation_dimension, "observation"),
    )
    for name in names:
        template = templates[name]
        data = jnp.concatenate(tuple(parts[name]), axis=0)
        expected = tuple(int(size) for size in template.data.shape)
        if tuple(data.shape[2:]) != expected:
            raise ValueError("Posterior observation shape changed between draws.")
        data = data.reshape((*first_shape, count, *expected))
        valid_data = jnp.all(
            jnp.isfinite(data).reshape((*first_shape, count, -1)),
            axis=-1,
        )
        _raise_invalid(valid_data, valid_policy, owner="Posterior observation")
        predictions[name] = PredictiveField(
            cx.Field(data, dims=(*all_dimensions, *template.dims)),
            sample_axes,
            valid=cx.Field(valid_data, dims=all_dimensions),
        )
    if single:
        return predictions[names[0]]
    return frozendict(predictions)


def _evaluate_position_callback(
    positions: PyTree[Array],
    callback: Any,
    /,
    *,
    sample_dims: tuple[str, ...],
    sample_sources: tuple[UncertaintySource, ...],
    batch_size: int | None,
    valid_policy: Literal["record", "raise"],
    owner: str,
) -> PredictiveField | frozendict[str, PredictiveField]:
    first_shape, total, flat_positions = _position_sample_layout(positions, sample_dims)
    chunk = _batch_size(batch_size, total)
    first_position = jax.tree_util.tree_map(lambda value: value[0], flat_positions)
    templates, single = _prediction_mapping(callback(first_position))
    names = tuple(templates)
    parts: dict[str, list[Array]] = {name: [] for name in names}

    def evaluate(position):
        result, result_single = _prediction_mapping(callback(position))
        if result_single != single or tuple(result) != names:
            raise ValueError(f"{owner} structure changed between draws.")
        return tuple(jnp.asarray(result[name].data) for name in names)

    for start in range(0, total, chunk):
        stop = min(start + chunk, total)
        selected = jax.tree_util.tree_map(lambda value: value[start:stop], flat_positions)
        evaluated = jax.vmap(evaluate)(selected)
        for name, data in zip(names, evaluated, strict=True):
            parts[name].append(jnp.asarray(data))

    predictions: dict[str, PredictiveField] = {}
    sample_axes = tuple(
        SampleAxis(dim, source)
        for dim, source in zip(sample_dims, sample_sources, strict=True)
    )
    for name in names:
        template = templates[name]
        data = jnp.concatenate(tuple(parts[name]), axis=0)
        expected = tuple(int(size) for size in template.data.shape)
        if tuple(data.shape[1:]) != expected:
            raise ValueError(f"{owner} shape changed between draws.")
        data = data.reshape((*first_shape, *expected))
        valid_data = jnp.all(jnp.isfinite(data).reshape((*first_shape, -1)), axis=-1)
        _raise_invalid(valid_data, valid_policy, owner=owner)
        predictions[name] = PredictiveField(
            cx.Field(data, dims=(*sample_dims, *template.dims)),
            sample_axes,
            valid=cx.Field(valid_data, dims=sample_dims),
        )
    if single:
        return predictions[names[0]]
    return frozendict(predictions)


def _sample_metadata(
    sample_dims: Sequence[str],
    sample_sources: Sequence[UncertaintySource],
) -> tuple[tuple[str, ...], tuple[UncertaintySource, ...]]:
    dimensions = tuple(sample_dims)
    sources = tuple(sample_sources)
    if not dimensions or len(dimensions) != len(sources):
        raise ValueError(
            "sample_dims and sample_sources must have equal non-zero length."
        )
    if len(set(dimensions)) != len(dimensions) or any(not dim for dim in dimensions):
        raise ValueError("sample_dims must be distinct non-empty strings.")
    return dimensions, sources


def _position_sample_layout(
    positions: PyTree[Array],
    dimensions: tuple[str, ...],
) -> tuple[tuple[int, ...], int, PyTree[Array]]:
    leaves = jax.tree_util.tree_leaves(positions)
    if not leaves:
        raise ValueError("Posterior position samples must contain array leaves.")
    axis_count = len(dimensions)
    first_shape = tuple(int(size) for size in leaves[0].shape[:axis_count])
    if len(first_shape) != axis_count or any(size <= 0 for size in first_shape):
        raise ValueError("Posterior sample axes must be present and non-empty.")
    for leaf in leaves:
        if tuple(int(size) for size in leaf.shape[:axis_count]) != first_shape:
            raise ValueError("Posterior position leaves have inconsistent sample axes.")
    total = 1
    for size in first_shape:
        total *= size
    flat_positions = jax.tree_util.tree_map(
        lambda value: jnp.asarray(value).reshape((total, *value.shape[axis_count:])),
        positions,
    )
    return first_shape, total, flat_positions


def _batch_size(value: int | None, total: int) -> int:
    chunk = total if value is None else int(value)
    if chunk <= 0:
        raise ValueError("batch_size must be positive.")
    return chunk


def _raise_invalid(
    valid: Array,
    policy: Literal["record", "raise"],
    /,
    *,
    owner: str,
) -> None:
    if policy not in ("record", "raise"):
        raise ValueError("valid_policy must be 'record' or 'raise'.")
    if policy == "raise" and not bool(jnp.all(valid)):
        failed = tuple(tuple(int(index) for index in row) for row in jnp.argwhere(~valid))
        raise FloatingPointError(f"{owner} produced invalid sample indices {failed!r}.")


def _attach_conditional_variance(
    predictions: PredictiveField | frozendict[str, PredictiveField],
    conditional: PredictiveField | frozendict[str, PredictiveField],
) -> PredictiveField | frozendict[str, PredictiveField]:
    prediction_map, prediction_single = _predictive_mapping(predictions)
    conditional_map, conditional_single = _predictive_mapping(conditional)
    if prediction_single != conditional_single or tuple(prediction_map) != tuple(
        conditional_map
    ):
        raise ValueError(
            "Conditional observation variance must match prediction structure."
        )
    result = {
        name: _with_conditional_variance(
            prediction_map[name],
            conditional_map[name],
        )
        for name in prediction_map
    }
    if prediction_single:
        return result["prediction"]
    return frozendict(result)


def _with_conditional_variance(
    prediction: PredictiveField,
    conditional: PredictiveField,
) -> PredictiveField:
    if prediction.sample_axes != conditional.sample_axes:
        raise ValueError(
            "Conditional observation variance sample axes must match predictions."
        )
    valid = prediction.valid
    if valid is not None and conditional.valid is not None:
        valid = cx.Field(
            jnp.asarray(valid.data) & jnp.asarray(conditional.valid.data),
            dims=valid.dims,
        )
    elif conditional.valid is not None:
        valid = conditional.valid
    return PredictiveField(
        prediction.samples,
        prediction.sample_axes,
        conditional_variance=conditional.samples,
        valid=valid,
    )


def _predictive_mapping(
    value: PredictiveField | Mapping[str, PredictiveField],
) -> tuple[dict[str, PredictiveField], bool]:
    if isinstance(value, PredictiveField):
        return {"prediction": value}, True
    result = dict(value)
    if not result or any(
        not isinstance(field, PredictiveField) for field in result.values()
    ):
        raise TypeError(
            "Expected a PredictiveField or non-empty PredictiveField mapping."
        )
    return result, False


def _prediction_mapping(value: Any) -> tuple[dict[str, cx.Field], bool]:
    if isinstance(value, cx.Field):
        return {"prediction": value}, True
    if not isinstance(value, Mapping) or not value:
        raise TypeError(
            "Posterior prediction must return a coordax.Field or non-empty field mapping."
        )
    result = dict(value)
    if any(not isinstance(name, str) or not name for name in result):
        raise TypeError("Posterior prediction labels must be non-empty strings.")
    if any(not isinstance(field, cx.Field) for field in result.values()):
        raise TypeError(
            "Every posterior prediction mapping value must be a coordax.Field."
        )
    return result, False


__all__ = [
    "predict_from_position_samples",
    "sample_observations_from_position_samples",
]
