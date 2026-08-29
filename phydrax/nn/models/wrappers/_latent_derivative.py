#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental.jet import jet

from ..._utils import _get_size


def jet_nth(fun, x: jax.Array, direction: jax.Array, /, *, order: int) -> jax.Array:
    order_i = int(order)
    if order_i < 1:
        raise ValueError("order must be positive.")
    x_arr = jnp.asarray(x)
    direction_arr = jnp.asarray(direction)
    zeros = jnp.zeros_like(x_arr)
    series = (tuple([direction_arr] + [zeros] * (order_i - 1)),)
    output = fun(x_arr)
    if jnp.iscomplexobj(output):
        _, real_terms = jet(lambda value: jnp.real(fun(value)), (x_arr,), series)
        _, imag_terms = jet(lambda value: jnp.imag(fun(value)), (x_arr,), series)
        return real_terms[-1] + 1j * imag_terms[-1]
    _, terms = jet(fun, (x_arr,), series)
    return terms[-1]


def _contract_grouped(
    model: Any,
    latents: Sequence[jax.Array],
    batch_shapes: Sequence[tuple[int, ...]],
    /,
) -> jax.Array:
    return model._contract_latents(latents, batch_shapes, topology="grouped")


def _contract_flat(
    model: Any,
    latents: Sequence[jax.Array],
    batch_shapes: Sequence[tuple[int, ...]],
    /,
) -> jax.Array:
    return model._contract_latents(latents, batch_shapes, topology="flat")


def _contraction_plan(model: Any, /):
    plan = model._plan_topology(supports_flat=True)
    executor = _contract_grouped if plan.effective == "grouped" else _contract_flat
    return executor, plan.fallback_message


def factor_nth_latents(
    model: Any,
    factor_model: Any,
    points: Any,
    /,
    *,
    name: str,
    key: Any,
    axis: int,
    order: int,
) -> tuple[jax.Array | None, tuple[int, ...] | None, str | None]:
    input_size = _get_size(factor_model.in_size)
    if not 0 <= int(axis) < int(input_size):
        return (
            None,
            None,
            f"requested axis={axis} but factor {name!r} has in_size={input_size}.",
        )

    if isinstance(points, tuple):
        coordinates = tuple(jnp.asarray(coordinate) for coordinate in points)
        if len(coordinates) != int(input_size):
            return (
                None,
                None,
                f"factor {name!r} expected {input_size} coord arrays, "
                f"got {len(coordinates)}.",
            )
        if not all(coordinate.ndim == 1 for coordinate in coordinates):
            return (
                None,
                None,
                f"factor {name!r} requires 1D coord arrays for tuple inputs.",
            )

        def latents_at_coordinate(coordinate):
            new_points = coordinates[:axis] + (coordinate,) + coordinates[axis + 1 :]
            latents, _ = model._eval_factor(
                factor_model,
                new_points,
                name=name,
                key=key,
            )
            return latents

        direction = jnp.ones_like(coordinates[axis])
        latents = jet_nth(
            latents_at_coordinate,
            coordinates[axis],
            direction,
            order=order,
        )
        return latents, tuple(int(value.shape[0]) for value in coordinates), None

    array = jnp.asarray(points)

    def latents_from_input(value):
        latents, _ = model._eval_factor_array(
            factor_model,
            value,
            name=name,
            key=key,
        )
        return latents

    if array.ndim == 0:
        if int(input_size) != 1:
            return (
                None,
                None,
                f"factor {name!r} scalar input is incompatible with "
                f"in_size={input_size}.",
            )
        latents = jet_nth(
            latents_from_input,
            array,
            jnp.ones_like(array),
            order=order,
        )
        return latents, (), None

    if array.ndim == 1:
        if int(input_size) == 1:
            direction = jnp.ones_like(array)
        elif int(array.shape[0]) == int(input_size):
            direction = jnp.zeros_like(array).at[axis].set(1.0)
        else:
            return (
                None,
                None,
                f"factor {name!r} expected shape ({input_size},), got {array.shape}.",
            )
        latents = jet_nth(
            latents_from_input,
            array,
            direction,
            order=order,
        )
        _, batch_shape = model._eval_factor_array(
            factor_model,
            array,
            name=name,
            key=key,
        )
        return latents, batch_shape, None

    if array.ndim == 2 and int(array.shape[1]) == int(input_size):

        def nth_single(row):
            direction = (
                jnp.ones_like(row)
                if int(input_size) == 1
                else jnp.zeros_like(row).at[axis].set(1.0)
            )
            return jet_nth(latents_from_input, row, direction, order=order)

        return jax.vmap(nth_single)(array), (int(array.shape[0]),), None

    return (
        None,
        None,
        f"factor {name!r} has unsupported input shape {array.shape} for optimized "
        "derivative evaluation.",
    )


def evaluate_latent_partial(
    model: Any,
    /,
    *,
    deps: tuple[str, ...],
    var: str,
    axis: int,
    order: int,
    args: tuple[Any, ...],
    key: Any,
    kwargs: dict[str, Any],
) -> tuple[jax.Array | None, str | None]:
    """Evaluate one optimized latent-factor partial derivative when supported."""
    del kwargs
    if len(args) != len(model.factor_models):
        return (
            None,
            "optimized latent derivative evaluation requires one dependency per "
            "latent factor.",
        )
    if var not in deps:
        return None, f"variable {var!r} is not in DomainFunction dependencies."

    if model.execution_policy.layout not in model._supported_layouts:
        model._auto_fallback(
            "Latent derivative evaluation supports "
            "auto/dense_points/coord_separable/hybrid/full_tensor layouts; "
            f"requested layout={model.execution_policy.layout!r}. Falling back to "
            "generic derivatives."
        )
        return None, None

    executor, fallback_message = _contraction_plan(model)
    if fallback_message is not None:
        model._auto_fallback(fallback_message)

    dependency_index = int(deps.index(var))
    keys = model._split_key(key)
    latents: list[jax.Array] = []
    batch_shapes: list[tuple[int, ...]] = []
    for index, (name, factor_model, points, factor_key) in enumerate(
        zip(model.factor_names, model.factor_models, args, keys, strict=True)
    ):
        if index == dependency_index:
            latent, batch_shape, reason = factor_nth_latents(
                model,
                factor_model,
                points,
                name=name,
                key=factor_key,
                axis=axis,
                order=order,
            )
            if reason is not None or latent is None or batch_shape is None:
                return None, reason
        else:
            latent, batch_shape = model._eval_factor(
                factor_model,
                points,
                name=name,
                key=factor_key,
            )
        latents.append(latent)
        batch_shapes.append(batch_shape)

    try:
        output = executor(model, latents, batch_shapes)
    except Exception as error:
        if model.execution_policy.topology != "best_effort_flat":
            return (
                None,
                "latent derivative provider failed to evaluate optimized path: "
                + str(error),
            )
        model._auto_fallback(
            "Latent derivative flat provider failed; falling back to grouped. "
            f"Reason: {error}"
        )
        output = _contract_grouped(model, latents, batch_shapes)
    output = model._finalize(output)

    batch_rank = sum(len(shape) for shape in batch_shapes)
    if batch_rank > 1 and output.ndim >= batch_rank:
        axes_by_dependency: list[tuple[int, ...]] = []
        cursor = 0
        for shape in batch_shapes:
            axes = tuple(range(cursor, cursor + len(shape)))
            axes_by_dependency.append(axes)
            cursor += len(shape)

        dense_axes: list[int] = []
        coordinate_axes: list[int] = []
        for dependency, axes in zip(args, axes_by_dependency, strict=True):
            if isinstance(dependency, tuple):
                coordinate_axes.extend(axes)
            else:
                dense_axes.extend(axes)

        desired_axes = tuple(dense_axes + coordinate_axes)
        current_axes = tuple(range(batch_rank))
        if desired_axes != current_axes:
            output = jnp.transpose(
                output,
                desired_axes + tuple(range(batch_rank, output.ndim)),
            )

    return output, None


__all__ = ["evaluate_latent_partial", "factor_nth_latents", "jet_nth"]
