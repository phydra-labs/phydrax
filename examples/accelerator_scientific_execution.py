# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Train a float32 Phydrax Poisson PINN on an explicitly selected accelerator.

Install phydrax[oneapi], phydrax[tpu], or phydrax[rocm10]
on supported Linux hardware. ROCm10 includes its userspace and both GPU architectures.
Pass the matching --accelerator selector and an explicit --device-id (default 0).
TPU multi-host launcher discovery is opt-in with --tpu-autodetect; this example
qualifies one selected local device in one process, not every visible device.
Other devices remain visible and are not claimed as qualified. No CPU fallback.
ROCm initializes CPU for JAX host callbacks, not for scientific array execution.
TPU highest matmul precision uses six BF16 passes, not native IEEE FP32.
oneAPI uses an experimental mixed JAX/JAXlib release stack.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from typing import Any, get_args, Literal, NamedTuple, TYPE_CHECKING, TypeAlias


if TYPE_CHECKING:
    from jax import Array, Device


Accelerator: TypeAlias = Literal["oneapi", "tpu", "rocm10"]


class PoissonTraining(NamedTuple):
    """Float32 PINN parameters, Adam state, and their pure residual/update."""

    parameters: Any
    state: Any
    residual: Callable[[Any], Array]
    objective: Callable[[Any], Array]
    step: Callable[[Any, Any], tuple[Any, Any, Array, Any]]


def poisson_training(*, device: Device) -> PoissonTraining:
    """Prepare a normal domain-bound MLP Poisson PINN and its Adam update step.

    The key, domain bounds, collocation points, parameters, and optimizer state
    are committed to `device`, so every residual, objective, and update derived
    from them runs on that one device without per-step transfers.
    """
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np

    import phydrax as phx
    from phydrax.axes import AxisArray
    from phydrax.operators.differential import laplacian
    from phydrax.optim import adam

    box = phx.domain.HyperRectangle(
        jax.device_put(np.zeros(2, np.float32), device),
        jax.device_put(np.ones(2, np.float32), device),
        dtype=jnp.float32,
    )
    model = phx.nn.models.MLP(
        in_size=2,
        out_size="scalar",
        width_size=8,
        depth=2,
        activation=jnp.tanh,
        dtype=jnp.float32,
        key=jax.device_put(jax.random.key(7), device),
    )
    points = jax.device_put(
        np.asarray([[0.2, 0.3], [0.4, 0.7], [0.7, 0.5], [0.8, 0.9]], np.float32),
        device,
    )
    parameters, static = eqx.partition(model, eqx.is_inexact_array)
    parameters = jax.device_put(parameters, device)
    optimizer = adam(1e-3)
    state = jax.device_put(optimizer.init(parameters), device)

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

    def step(current: Any, current_state: Any) -> tuple[Any, Any, jax.Array, Any]:
        value, gradient = eqx.filter_value_and_grad(objective)(current)
        updates, next_state = optimizer.update(gradient, current_state, current)
        return eqx.apply_updates(current, updates), next_state, value, gradient

    return PoissonTraining(parameters, state, residual, objective, step)


def poisson_steps(
    *, device: Device, steps: int = 20
) -> tuple[dict[str, float], tuple[Array, ...]]:
    """Differentiate a normal domain-bound MLP twice in space and train it.

    Training, every compiled step, and every returned array run on `device`,
    where `poisson_training` commits the prepared PINN.
    """
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np

    if steps < 1:
        raise ValueError("steps must be positive")
    parameters, state, residual, objective, update = poisson_training(device=device)
    step = eqx.filter_jit(update)

    initial, initial_gradient = eqx.filter_value_and_grad(objective)(parameters)
    directional = jnp.sqrt(
        sum(jnp.sum(value**2) for value in jax.tree.leaves(initial_gradient))
    )
    direction = jax.tree.map(lambda value: value / directional, initial_gradient)
    plus = jax.tree.map(
        lambda value, delta: value + np.float32(0.01) * delta, parameters, direction
    )
    minus = jax.tree.map(
        lambda value, delta: value - np.float32(0.01) * delta, parameters, direction
    )
    finite_difference = (objective(plus) - objective(minus)) / np.float32(0.02)
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


def selected_device_record(
    accelerator: Accelerator,
    *,
    device_id: int = 0,
    steps: int = 20,
    tpu_autodetect: bool = False,
) -> dict[str, Any]:
    """Configure the runtime, train on the selected device, and return its record.

    An unavailable accelerator or device raises; a completed run that misses an
    acceptance criterion returns ``status="failed"``.
    """
    import phydrax as phx

    accelerator = phx.typing.parse(accelerator, Accelerator, "accelerator")
    if tpu_autodetect and accelerator != "tpu":
        raise ValueError("TPU autodetection requires the tpu accelerator")
    if device_id < 0:
        raise ValueError("device_id must be nonnegative")
    is_rocm = accelerator == "rocm10"
    runtime_platform = "rocm" if is_rocm else accelerator
    bootstrap = phx.config.RuntimeBootstrap(autodetect=True) if tpu_autodetect else None
    phx.config.configure(
        phx.config.RuntimeConfig(
            platforms=("rocm", "cpu") if is_rocm else (runtime_platform,),
            enable_x64=True,
            bootstrap=bootstrap,
        )
    )
    import jax
    import numpy as np

    from phydrax.backends import (
        frontend_index_patch_identity,
        oneapi_environment,
        rocm_environment,
        tpu_environment,
    )

    match accelerator:
        case "oneapi":
            report = oneapi_environment(initialize_backend=True)
        case "tpu":
            report = tpu_environment(initialize_backend=True)
        case "rocm10":
            report = rocm_environment(initialize_backend=True)
        case _:
            raise ValueError("Unsupported accelerator")
    if not report.successful or not report.facts.device_probed:
        raise RuntimeError(f"Selected accelerator is unavailable: {report.issues}")
    devices = jax.local_devices(backend=runtime_platform)
    if not devices or jax.process_count() != 1:
        raise RuntimeError(
            "This example requires one process with an actual local selected device"
        )
    matching = [device for device in devices if device.id == device_id]
    if len(matching) != 1:
        raise RuntimeError(f"Device ID {device_id} is not a unique local device")
    device = matching[0]
    values, arrays = poisson_steps(device=device, steps=steps)
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
    return {
        "accelerator": accelerator,
        "platform": device.platform,
        "platform_version": device.client.platform_version,
        "device_kind": device.device_kind,
        "qualified_device_id": device.id,
        "execution_device_count": 1,
        "global_device_count": jax.device_count(runtime_platform),
        "local_device_count": jax.local_device_count(runtime_platform),
        "frontend_index_patch": frontend_index_patch_identity(),
        "results_on_device": placed,
        "steps": steps,
        "values": values,
        "status": "passed" if passed else "failed",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accelerator", required=True, choices=get_args(Accelerator))
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--tpu-autodetect", action="store_true")
    args = parser.parse_args()
    if args.tpu_autodetect and args.accelerator != "tpu":
        parser.error("--tpu-autodetect requires --accelerator tpu")
    if args.device_id < 0:
        parser.error("--device-id must be nonnegative")
    record = selected_device_record(
        args.accelerator,
        device_id=args.device_id,
        steps=args.steps,
        tpu_autodetect=args.tpu_autodetect,
    )
    print(json.dumps(record, sort_keys=True))
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
