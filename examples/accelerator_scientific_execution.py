# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Train a float32 Phydrax Poisson PINN on an explicitly selected accelerator.

Install phydrax[oneapi] on a supported Intel GPU Linux host or phydrax[tpu]
on a provisioned Linux TPU worker. Run with --accelerator oneapi or tpu.
TPU multi-host launcher discovery is opt-in with --tpu-autodetect; this example
qualifies one selected local device in one process, not every visible device.
Other devices remain visible and are not claimed as qualified. No CPU fallback.
TPU highest matmul precision uses six BF16 passes, not native IEEE FP32.
oneAPI uses an experimental mixed JAX/JAXlib release stack.
"""

from __future__ import annotations

import argparse
import json
from typing import Any


def poisson_steps(*, steps: int = 20) -> tuple[dict[str, float], tuple[Any, ...]]:
    """Differentiate a normal domain-bound MLP twice in space and train it."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np

    import phydrax as phx
    from phydrax.axes import AxisArray
    from phydrax.operators.differential import laplacian
    from phydrax.optim import adam

    if steps < 1:
        raise ValueError("steps must be positive")
    box = phx.domain.HyperRectangle(
        np.zeros(2, np.float32), np.ones(2, np.float32), dtype=jnp.float32
    )
    model = phx.nn.models.MLP(
        in_size=2,
        out_size="scalar",
        width_size=8,
        depth=2,
        activation=jnp.tanh,
        dtype=jnp.float32,
        key=jax.random.key(7),
    )
    points = jnp.asarray(
        [[0.2, 0.3], [0.4, 0.7], [0.7, 0.5], [0.8, 0.9]], dtype=jnp.float32
    )
    parameters, static = eqx.partition(model, eqx.is_inexact_array)
    optimizer = adam(1e-3)
    state = optimizer.init(parameters)

    def residual(candidate: Any) -> jax.Array:
        field = box.Model("x")(eqx.combine(candidate, static))
        curvature = laplacian(field, var="x")(
            {"x": AxisArray(points, dims=("point", None))}
        ).data
        forcing = (
            2
            * jnp.pi**2
            * jnp.sin(jnp.pi * points[:, 0])
            * jnp.sin(jnp.pi * points[:, 1])
        )
        return curvature + forcing

    def objective(candidate: Any) -> jax.Array:
        return jnp.mean(residual(candidate) ** 2)

    @eqx.filter_jit
    def step(current: Any, current_state: Any) -> tuple[Any, Any, jax.Array, Any]:
        value, gradient = eqx.filter_value_and_grad(objective)(current)
        updates, next_state = optimizer.update(gradient, current_state, current)
        return eqx.apply_updates(current, updates), next_state, value, gradient

    initial, initial_gradient = eqx.filter_value_and_grad(objective)(parameters)
    directional = jnp.sqrt(
        sum(jnp.sum(value**2) for value in jax.tree.leaves(initial_gradient))
    )
    direction = jax.tree.map(lambda value: value / directional, initial_gradient)
    plus = jax.tree.map(
        lambda value, delta: value + jnp.float32(0.01) * delta, parameters, direction
    )
    minus = jax.tree.map(
        lambda value, delta: value - jnp.float32(0.01) * delta, parameters, direction
    )
    finite_difference = (objective(plus) - objective(minus)) / jnp.float32(0.02)
    gradient = initial_gradient
    trained = parameters
    for _ in range(steps):
        trained, state, _, gradient = step(trained, state)
    final = objective(trained)
    gradient_norm = jnp.sqrt(
        sum(jnp.sum(value**2) for value in jax.tree.leaves(gradient))
    )
    update_norm = jnp.sqrt(
        sum(
            jnp.sum((new - old) ** 2)
            for new, old in zip(
                jax.tree.leaves(trained), jax.tree.leaves(parameters), strict=True
            )
        )
    )
    spatial = residual(trained)
    arrays = (
        initial,
        final,
        gradient_norm,
        update_norm,
        spatial,
        directional,
        finite_difference,
        *jax.tree.leaves(trained),
    )
    jax.block_until_ready(arrays)
    return {
        "initial_loss": float(initial),
        "final_loss": float(final),
        "gradient_norm": float(gradient_norm),
        "update_norm": float(update_norm),
        "directional_derivative": float(directional),
        "finite_difference_derivative": float(finite_difference),
    }, arrays


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accelerator", required=True, choices=("oneapi", "tpu"))
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--tpu-autodetect", action="store_true")
    args = parser.parse_args()
    import phydrax as phx

    if args.tpu_autodetect and args.accelerator != "tpu":
        parser.error("--tpu-autodetect requires --accelerator tpu")
    bootstrap = (
        phx.config.RuntimeBootstrap(autodetect=True) if args.tpu_autodetect else None
    )
    phx.config.configure(
        phx.config.RuntimeConfig(
            platforms=(args.accelerator,), enable_x64=True, bootstrap=bootstrap
        )
    )
    import jax
    import numpy as np

    from phydrax._jax_config import default_device
    from phydrax.backends import oneapi_environment, tpu_environment

    match args.accelerator:
        case "oneapi":
            report = oneapi_environment(initialize_backend=True)
        case "tpu":
            report = tpu_environment(initialize_backend=True)
        case _:
            raise ValueError("Unsupported accelerator")
    if not report.successful or not report.facts.device_probed:
        raise RuntimeError(f"Selected accelerator is unavailable: {report.issues}")
    devices = jax.local_devices(backend=args.accelerator)
    if not devices or jax.process_count() != 1:
        raise RuntimeError(
            "This example requires one process with an actual local selected device"
        )
    device = devices[0]
    with default_device(device):
        values, arrays = poisson_steps(steps=args.steps)
    placed = all(
        array.devices() == {device} and array.dtype == np.dtype("float32")
        for array in arrays
    )
    passed = (
        placed
        and all(np.isfinite(value) for value in values.values())
        and values["gradient_norm"] > 0
        and values["update_norm"] > 0
        and values["final_loss"] < values["initial_loss"]
    )
    passed = passed and bool(
        np.isclose(
            values["directional_derivative"],
            values["finite_difference_derivative"],
            rtol=0.03,
            atol=0.02,
        )
    )
    print(
        json.dumps(
            {
                "accelerator": args.accelerator,
                "platform": device.platform,
                "platform_version": device.client.platform_version,
                "device_kind": device.device_kind,
                "qualified_device_id": device.id,
                "execution_device_count": 1,
                "global_device_count": jax.device_count(args.accelerator),
                "local_device_count": jax.local_device_count(args.accelerator),
                "results_on_device": placed,
                "steps": args.steps,
                "values": values,
                "status": "passed" if passed else "failed",
            },
            sort_keys=True,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
