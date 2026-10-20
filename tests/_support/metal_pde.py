#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded float32 Poisson PINN workload shared by a Metal case and its CPU reference.

The workload solves ``-Δu = 2π² sin(πx) sin(πy)`` on the unit square with
``u = 0`` on its boundary, using an explicit float32 `MLP` bound to a float32
`HyperRectangle` and the Phydrax coordinate `laplacian`. Collocation points are
generated once on the host and rounded once to the workload dtype.

Run as ``python -m tests._support.metal_pde INPUT OUTPUT`` in a separate CPU
process to compute the float64 reference of a device step from the same float32
parameters and points (see `write_reference`).
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

import phydrax as phx
import phydrax.axes as cx
from phydrax.operators.differential import laplacian
from phydrax.optim import adam


WIDTH = 16
DEPTH = 2
LEARNING_RATE = 1e-3
MODEL_SEED = 0
_GRID = 8


def poisson_points(dtype: Any, /) -> tuple[np.ndarray, np.ndarray]:
    """Interior midpoint grid and boundary points, rounded once to `dtype`."""
    nodes = (np.arange(_GRID, dtype=np.float64) + 0.5) / _GRID
    x, y = np.meshgrid(nodes, nodes, indexing="ij")
    interior = np.stack((x.ravel(), y.ravel()), axis=-1)
    zeros, ones = np.zeros_like(nodes), np.ones_like(nodes)
    boundary = np.concatenate(
        (
            np.stack((nodes, zeros), axis=-1),
            np.stack((nodes, ones), axis=-1),
            np.stack((zeros, nodes), axis=-1),
            np.stack((ones, nodes), axis=-1),
        )
    )
    return interior.astype(dtype), boundary.astype(dtype)


def unit_square(dtype: Any, /) -> phx.domain.HyperRectangle:
    return phx.domain.HyperRectangle(np.zeros(2), np.ones(2), dtype=dtype)


def build_model() -> phx.nn.models.MLP:
    return phx.nn.models.MLP(
        in_size=2,
        out_size="scalar",
        width_size=WIDTH,
        depth=DEPTH,
        activation=jnp.tanh,
        dtype=jnp.float32,
        key=jr.key(MODEL_SEED),
    )


def point_mapping(values: jax.Array, /) -> dict[str, cx.AxisArray]:
    """Evaluation points for the unit-square geometry label `x`."""
    return {"x": cx.AxisArray(values, dims=("point", None))}


def poisson_loss(
    model: Any,
    box: phx.domain.HyperRectangle,
    interior: jax.Array,
    boundary: jax.Array,
    /,
) -> jax.Array:
    """Mean squared PDE residual plus the mean squared boundary value."""
    field = box.Model("x")(model)
    x, y = interior[:, 0], interior[:, 1]
    forcing = 2.0 * jnp.pi**2 * jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)
    residual = laplacian(field, var="x")(point_mapping(interior)).data + forcing
    boundary_values = field(point_mapping(boundary)).data
    return jnp.mean(residual**2) + jnp.mean(boundary_values**2)


def optimizer() -> optax.GradientTransformation:
    return adam(LEARNING_RATE)


def train_step(
    box: phx.domain.HyperRectangle,
    static: Any,
    interior: jax.Array,
    boundary: jax.Array,
    /,
) -> Callable[[Any, Any], tuple[jax.Array, Any, Any, Any]]:
    """Compiled loss, parameter gradient and Adam update for fixed collocation."""
    transform = optimizer()

    @eqx.filter_jit
    def step(parameters: Any, optimizer_state: Any) -> tuple[jax.Array, Any, Any, Any]:
        def objective(candidate: Any) -> jax.Array:
            return poisson_loss(eqx.combine(candidate, static), box, interior, boundary)

        loss, gradient = eqx.filter_value_and_grad(objective)(parameters)
        updates, next_state = transform.update(gradient, optimizer_state, parameters)
        return loss, gradient, eqx.apply_updates(parameters, updates), next_state

    return step


def _leaves(tree: Any, /) -> list[Any]:
    return jax.tree.leaves(tree)


def write_reference(input_path: str, output_path: str, /) -> None:
    """CPU reference: float64 loss/gradient and float32 Adam of the device gradient."""
    data = np.load(input_path)
    parameters, static = eqx.partition(build_model(), eqx.is_inexact_array)
    definition = jax.tree.structure(parameters)
    count = definition.num_leaves
    device_parameters = jax.tree.unflatten(
        definition, [jnp.asarray(data[f"parameter_{i}"]) for i in range(count)]
    )
    device_gradient = jax.tree.unflatten(
        definition, [jnp.asarray(data[f"gradient_{i}"]) for i in range(count)]
    )
    interior, boundary = data["interior"], data["boundary"]

    reference_parameters = jax.tree.map(
        lambda leaf: leaf.astype(jnp.float64), device_parameters
    )
    reference_loss, reference_gradient = eqx.filter_value_and_grad(
        lambda candidate: poisson_loss(
            eqx.combine(candidate, static),
            unit_square(jnp.float64),
            jnp.asarray(interior, dtype=jnp.float64),
            jnp.asarray(boundary, dtype=jnp.float64),
        )
    )(reference_parameters)

    transform = optimizer()
    updates, _ = transform.update(
        device_gradient, transform.init(device_parameters), device_parameters
    )
    expected = eqx.apply_updates(device_parameters, updates)

    arrays: dict[str, np.ndarray] = {"loss": np.asarray(reference_loss)}
    for index, leaf in enumerate(_leaves(reference_gradient)):
        arrays[f"gradient_{index}"] = np.asarray(leaf)
    for index, leaf in enumerate(_leaves(expected)):
        arrays[f"updated_{index}"] = np.asarray(leaf)
    np.savez(output_path, allow_pickle=False, **arrays)


if __name__ == "__main__":
    if not phx.config.x64_enabled() or jax.default_backend() != "cpu":
        raise SystemExit("The Poisson reference must run on CPU with x64 enabled.")
    write_reference(sys.argv[1], sys.argv[2])
