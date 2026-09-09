#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Qualify tangent spherical calculus and report separately synchronized timings.

Run from the repository root with ``python -m tools.spherical_vector_qualification``.
Source projection and untimed reference calculations are outside operation timings.
No benchmark result or release qualification is assumed until this script runs.
"""

from __future__ import annotations

import argparse
import json
import time

import equinox as eqx
import jax
import jax.numpy as jnp

from phydrax.discretization import SphericalSpectralPlan
from phydrax.discretization.spectral._spherical_vector import (
    PreparedSphericalVectorOperators,
)


def _measure(function, arguments, repeats):
    started = time.perf_counter()
    lowered = eqx.filter_jit(function).lower(*arguments)
    trace_ms = 1e3 * (time.perf_counter() - started)
    started = time.perf_counter()
    compiled = lowered.compile()
    compile_ms = 1e3 * (time.perf_counter() - started)
    started = time.perf_counter()
    result = jax.block_until_ready(compiled(*arguments))
    first_execution_ms = 1e3 * (time.perf_counter() - started)
    durations = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = jax.block_until_ready(compiled(*arguments))
        durations.append(1e3 * (time.perf_counter() - started))
    return result, {
        "trace_ms": trace_ms,
        "compile_ms": compile_ms,
        "first_execution_ms": first_execution_ms,
        "execution_mean_ms": sum(durations) / repeats,
        "execution_min_ms": min(durations),
    }


def run_qualification(
    bandlimit=8, *, sampling="mwss", execution="recursive", radius=2.0, repeats=5
):
    if bandlimit < 3 or repeats < 1:
        raise ValueError("Qualification requires bandlimit >= 3 and repeats >= 1.")
    started = time.perf_counter()
    space = SphericalSpectralPlan(
        bandlimit, sampling=sampling, execution=execution
    ).prepare(radius=radius)
    jax.block_until_ready(space)
    scalar_prepare_ms = 1e3 * (time.perf_counter() - started)
    started = time.perf_counter()
    vector = PreparedSphericalVectorOperators(space)
    jax.block_until_ready(vector)
    vector_prepare_ms = 1e3 * (time.perf_counter() - started)
    theta, phi = jnp.meshgrid(space.transform.theta, space.transform.phi, indexing="ij")
    scalar = jnp.sin(theta) * jnp.cos(phi)
    coefficients = space.project(scalar)
    second = space.project(jnp.sin(theta) ** 2 * jnp.cos(2.0 * phi))
    zero = jnp.zeros_like(coefficients)
    vorticity = space.modal_laplacian(coefficients)
    divergence = space.modal_laplacian(second)
    gradient, gradient_timing = _measure(vector.gradient, (coefficients,), repeats)
    east, north = gradient
    div_gradient, div_timing = _measure(vector.divergence, gradient, repeats)
    curl_gradient, curl_timing = _measure(vector.curl, gradient, repeats)
    wind, wind_timing = _measure(vector.wind, (vorticity, divergence), repeats)
    omega = 0.23
    rotation_east = (
        omega * radius * (0.3 * jnp.sin(theta) - jnp.cos(theta) * jnp.cos(phi))
    )
    rotation_north = omega * radius * jnp.sin(phi)
    rotation_vorticity = space.project(
        2.0 * omega * (jnp.sin(theta) * jnp.cos(phi) + 0.3 * jnp.cos(theta))
    )
    rotation_wind = vector.wind(rotation_vorticity, zero)
    constant = zero.at[0, bandlimit - 1].set(1.0)
    errors = {
        "analytic_east_gradient": jnp.max(jnp.abs(east + jnp.sin(phi) / radius)),
        "analytic_north_gradient": jnp.max(
            jnp.abs(north + jnp.cos(theta) * jnp.cos(phi) / radius)
        ),
        "div_gradient_laplacian": jnp.max(jnp.abs(div_gradient - vorticity)),
        "curl_gradient": jnp.max(jnp.abs(curl_gradient)),
        "div_rotated_gradient": jnp.max(jnp.abs(vector.divergence(-north, east))),
        "curl_rotated_gradient_laplacian": jnp.max(
            jnp.abs(vector.curl(-north, east) - vorticity)
        ),
        "helmholtz_divergence": jnp.max(jnp.abs(vector.divergence(*wind) - divergence)),
        "helmholtz_vorticity": jnp.max(jnp.abs(vector.curl(*wind) - vorticity)),
        "solid_rotation_divergence": jnp.max(
            jnp.abs(vector.divergence(rotation_east, rotation_north))
        ),
        "solid_rotation_curl": jnp.max(
            jnp.abs(vector.curl(rotation_east, rotation_north) - rotation_vorticity)
        ),
        "solid_rotation_wind_east": jnp.max(jnp.abs(rotation_wind[0] - rotation_east)),
        "solid_rotation_wind_north": jnp.max(jnp.abs(rotation_wind[1] - rotation_north)),
        "constant_gradient": jnp.max(jnp.abs(jnp.stack(vector.gradient(constant)))),
    }
    errors = {name: float(value) for name, value in errors.items()}
    tolerance = 1e-9
    finite = all(bool(jnp.isfinite(value)) for value in errors.values())
    return {
        "operator_id": vector.operator_id,
        "bandlimit": bandlimit,
        "sampling": sampling,
        "execution": execution,
        "radius": radius,
        "repeats": repeats,
        "x64_enabled": jax.config.x64_enabled,
        "backend": jax.default_backend(),
        "sample_shape": space.sample_shape,
        "sampled_poles": int(
            jnp.count_nonzero(jnp.abs(jnp.sin(space.transform.theta)) < 1e-14)
        ),
        "scalar_transform_precompute_bytes": space.transform.precompute_bytes,
        "spin_transform_precompute_bytes": vector.raise_operator.output_transform.precompute_bytes,
        "scalar_prepare_ms": scalar_prepare_ms,
        "vector_prepare_ms": vector_prepare_ms,
        "operations": {
            "gradient": gradient_timing,
            "divergence": div_timing,
            "curl": curl_timing,
            "wind": wind_timing,
        },
        "absolute_errors": errors,
        "absolute_tolerance": tolerance,
        "null_mode_defect": float(vector.null_mode_defect(constant)),
        "passed": finite and all(value <= tolerance for value in errors.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bandlimit", type=int, default=8)
    parser.add_argument("--sampling", choices=("mw", "mwss", "dh", "gl"), default="mwss")
    parser.add_argument(
        "--execution", choices=("recursive", "precomputed"), default="recursive"
    )
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--repeats", type=int, default=5)
    arguments = parser.parse_args()
    result = run_qualification(**vars(arguments))
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
