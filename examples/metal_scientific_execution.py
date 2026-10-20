#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Float32 Poisson PINN trained on the Apple GPU (Metal) with x64 enabled.

Solves ``-Δu = 2π² sin(πx) sin(πy)`` on the unit square with ``u = 0`` on the
boundary (exact solution ``sin(πx) sin(πy)``). The trial field is an explicit
float32 `MLP` bound to a float32 `HyperRectangle`; the residual uses the Phydrax
coordinate `laplacian` of that domain field. Each compiled step evaluates the
residual and boundary loss, its parameter gradient, and one Adam update.

The script selects the ``mps`` platform before any numerical work, verifies the
installed companion and the probed device, and checks that every reported result
array lives on that device in float32. It never falls back to CPU. The last line
of standard output is one JSON record with the device identity, the numerical
results and a ``status``; the exit code is nonzero unless the status passed.

Install ``phydrax[metal]`` on Apple silicon, then run::

    python examples/metal_scientific_execution.py
"""

import json
import math
import time
from typing import Any

import phydrax as phx


# Select the Apple GPU before any numerical module creates an array.
phx.config.configure(phx.config.RuntimeConfig(platforms=("mps",), enable_x64=True))

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import phydrax.axes as cx
from phydrax.backends import metal_environment
from phydrax.operators.differential import laplacian
from phydrax.optim import adam


WIDTH = 32
DEPTH = 3
STEPS = 400
LEARNING_RATE = 2e-3
COLLOCATION_GRID = 16
EVALUATION_GRID = 33


def _metal_device() -> jax.Device:
    report = metal_environment()
    if not (report.successful and report.facts.device_probed):
        issues = "; ".join(f"{issue.code}: {issue.message}" for issue in report.issues)
        raise SystemExit(f"Apple GPU execution is unavailable: {issues or 'unprobed'}")
    if not phx.config.x64_enabled():
        raise SystemExit("This workflow requires x64 enabled.")
    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "mps":
        raise SystemExit(f"Expected exactly one selected MPS device, found {devices}.")
    return devices[0]


def _collocation() -> tuple[np.ndarray, np.ndarray]:
    """Interior midpoint grid and boundary points, rounded once to float32."""
    nodes = (np.arange(COLLOCATION_GRID, dtype=np.float64) + 0.5) / COLLOCATION_GRID
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
    return interior.astype(np.float32), boundary.astype(np.float32)


def _points(values: jax.Array) -> dict[str, cx.AxisArray]:
    return {"x": cx.AxisArray(values, dims=("point", None))}


def _exact(points: jax.Array) -> jax.Array:
    return jnp.sin(jnp.pi * points[:, 0]) * jnp.sin(jnp.pi * points[:, 1])


def _loss(model: Any, box: Any, interior: jax.Array, boundary: jax.Array) -> jax.Array:
    field = box.Model("x")(model)
    forcing = 2.0 * jnp.pi**2 * _exact(interior)
    residual = laplacian(field, var="x")(_points(interior)).data + forcing
    boundary_values = field(_points(boundary)).data
    return jnp.mean(residual**2) + jnp.mean(boundary_values**2)


@eqx.filter_jit
def _norm(tree: Any) -> jax.Array:
    return jnp.sqrt(sum(jnp.sum(leaf**2) for leaf in jax.tree.leaves(tree)))


def main() -> int:
    device = _metal_device()
    host_interior, host_boundary = _collocation()
    interior, boundary = jnp.asarray(host_interior), jnp.asarray(host_boundary)
    nodes = np.linspace(0.0, 1.0, EVALUATION_GRID)
    grid = np.stack([axis.ravel() for axis in np.meshgrid(nodes, nodes)], axis=-1)
    evaluation = jnp.asarray(grid.astype(np.float32))

    box = phx.domain.HyperRectangle(
        np.zeros(2, np.float32), np.ones(2, np.float32), dtype=jnp.float32
    )
    model = phx.nn.models.MLP(
        in_size=2,
        out_size="scalar",
        width_size=WIDTH,
        depth=DEPTH,
        activation=jnp.tanh,
        dtype=jnp.float32,
        key=jr.key(0),
    )
    parameters, static = eqx.partition(model, eqx.is_inexact_array)
    optimizer = adam(LEARNING_RATE)
    optimizer_state = optimizer.init(parameters)

    @eqx.filter_jit
    def step(current: Any, state: Any) -> tuple[jax.Array, Any, Any, Any]:
        def objective(candidate: Any) -> jax.Array:
            return _loss(eqx.combine(candidate, static), box, interior, boundary)

        value, gradient = eqx.filter_value_and_grad(objective)(current)
        updates, next_state = optimizer.update(gradient, state, current)
        return value, gradient, eqx.apply_updates(current, updates), next_state

    @eqx.filter_jit
    def evaluate(current: Any) -> tuple[jax.Array, jax.Array]:
        field = box.Model("x")(eqx.combine(current, static))
        prediction = field(_points(evaluation)).data
        exact = _exact(evaluation)
        error = jnp.linalg.norm(prediction - exact) / jnp.linalg.norm(exact)
        loss = _loss(eqx.combine(current, static), box, interior, boundary)
        return loss, error

    started = time.perf_counter()
    initial_loss, gradient, trained, optimizer_state = step(parameters, optimizer_state)
    jax.block_until_ready(trained)
    first_step_seconds = time.perf_counter() - started
    first_update = _norm(jax.tree.map(lambda new, old: new - old, trained, parameters))
    gradient_norm = _norm(gradient)

    started = time.perf_counter()
    for _ in range(STEPS - 1):
        _, _, trained, optimizer_state = step(trained, optimizer_state)
    jax.block_until_ready(trained)
    steady_seconds = time.perf_counter() - started

    initial_evaluation = evaluate(parameters)
    final_loss, relative_l2 = evaluate(trained)
    results = (
        initial_loss,
        first_update,
        gradient_norm,
        final_loss,
        relative_l2,
        *initial_evaluation,
        *jax.tree.leaves(trained),
    )
    on_device = all(
        array.devices() == {device} and array.dtype == jnp.float32 for array in results
    )
    # Report the device that actually holds the trained result, not the request.
    (result_device,) = final_loss.devices()
    values = {
        "initial_loss": float(initial_loss),
        "final_loss": float(final_loss),
        "initial_relative_l2": float(initial_evaluation[1]),
        "final_relative_l2": float(relative_l2),
        "first_gradient_norm": float(gradient_norm),
        "first_update_norm": float(first_update),
    }
    passed = (
        on_device
        and all(math.isfinite(value) for value in values.values())
        and values["first_update_norm"] > 0.0
        and values["final_loss"] < values["initial_loss"]
        and values["final_relative_l2"] < values["initial_relative_l2"]
    )
    record = {
        "workflow": "metal-float32-poisson-pinn",
        "device": {
            "platform": result_device.platform,
            "platform_version": result_device.client.platform_version,
            "device_kind": result_device.device_kind,
        },
        "x64_enabled": bool(phx.config.x64_enabled()),
        "dtype": "float32",
        "results_on_device": on_device,
        "steps": STEPS,
        "learning_rate": LEARNING_RATE,
        "parameter_count": sum(leaf.size for leaf in jax.tree.leaves(parameters)),
        **values,
        "timing_seconds": {
            "first_step_including_compile": first_step_seconds,
            "remaining_steps": steady_seconds,
        },
        "status": "passed" if passed else "failed",
    }
    print(json.dumps(record, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
