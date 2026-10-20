#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Float32 Poisson PINN step on the selected Apple GPU (device file).

The process starts with ``JAX_PLATFORMS=mps`` and ``JAX_ENABLE_X64=1``. The
float64 reference runs in a separate CPU process on the same float32 parameters
and collocation points; no CPU computation happens in this process.
"""

import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from tests._support.metal_pde import (
    build_model,
    optimizer,
    poisson_loss,
    poisson_points,
    train_step,
    unit_square,
)
from tests._support.metal_qualification import metal_device, record_metal_execution


_ROOT = Path(__file__).resolve().parents[2]
# float32 unit roundoff is 6e-8; 1e-4 allows a 1e3 amplification through the
# third-order chain (Laplacian, then parameter gradient) of a depth-two network.
_FLOAT32_RELATIVE_BOUND = 1e-4


def _relative_error(actual: list[np.ndarray], reference: list[np.ndarray]) -> float:
    actual_flat = np.concatenate([np.ravel(leaf).astype(np.float64) for leaf in actual])
    reference_flat = np.concatenate([np.ravel(leaf) for leaf in reference])
    return float(
        np.linalg.norm(actual_flat - reference_flat) / np.linalg.norm(reference_flat)
    )


def _cpu_reference(
    tmp_path: Path,
    parameters: list[np.ndarray],
    gradient: list[np.ndarray],
    interior: np.ndarray,
    boundary: np.ndarray,
) -> Any:
    request, response = tmp_path / "device-step.npz", tmp_path / "reference.npz"
    arrays = {"interior": interior, "boundary": boundary}
    arrays.update({f"parameter_{i}": leaf for i, leaf in enumerate(parameters)})
    arrays.update({f"gradient_{i}": leaf for i, leaf in enumerate(gradient)})
    np.savez(request, allow_pickle=False, **arrays)
    environment = phx.config.environment(
        phx.config.RuntimeConfig(platforms=("cpu",), enable_x64=True)
    )
    result = subprocess.run(
        [sys.executable, "-m", "tests._support.metal_pde", str(request), str(response)],
        cwd=_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return np.load(response)


def test_float32_mlp_poisson_step_matches_cpu_reference(
    record_property: Callable[[str, object], None], tmp_path: Path
) -> None:
    device = metal_device()
    host_interior, host_boundary = poisson_points(np.float32)
    interior, boundary = jnp.asarray(host_interior), jnp.asarray(host_boundary)
    box = unit_square(jnp.float32)
    parameters, static = eqx.partition(build_model(), eqx.is_inexact_array)
    step = train_step(box, static, interior, boundary)
    optimizer_state = optimizer().init(parameters)

    loss, gradient, updated, _ = step(parameters, optimizer_state)
    next_loss = eqx.filter_jit(poisson_loss)(
        eqx.combine(updated, static), box, interior, boundary
    )
    jax.block_until_ready((loss, gradient, updated, next_loss))

    parameter_leaves = jax.tree.leaves(parameters)
    gradient_leaves = jax.tree.leaves(gradient)
    updated_leaves = jax.tree.leaves(updated)
    for leaf in (loss, next_loss, *parameter_leaves, *gradient_leaves, *updated_leaves):
        assert leaf.dtype == jnp.float32
        assert leaf.devices() == {device}
    assert bool(jnp.isfinite(loss)) and bool(jnp.isfinite(next_loss))
    assert float(next_loss) < float(loss)

    reference = _cpu_reference(
        tmp_path,
        [np.asarray(leaf) for leaf in parameter_leaves],
        [np.asarray(leaf) for leaf in gradient_leaves],
        host_interior,
        host_boundary,
    )
    reference_loss = float(reference["loss"])
    assert abs(float(loss) - reference_loss) <= _FLOAT32_RELATIVE_BOUND * reference_loss
    reference_gradient = [reference[f"gradient_{i}"] for i in range(len(gradient_leaves))]
    assert (
        _relative_error(
            [np.asarray(leaf) for leaf in gradient_leaves], reference_gradient
        )
        <= _FLOAT32_RELATIVE_BOUND
    )
    # The device Adam arithmetic matches the CPU float32 update of the same
    # device gradient up to float32 rounding of the parameter sum.
    for index, leaf in enumerate(updated_leaves):
        np.testing.assert_allclose(
            np.asarray(leaf), reference[f"updated_{index}"], rtol=2e-6, atol=1e-9
        )
    assert any(
        bool(jnp.any(new != old))
        for new, old in zip(updated_leaves, parameter_leaves, strict=True)
    )

    record_metal_execution(record_property, loss, next_loss, *updated_leaves)
