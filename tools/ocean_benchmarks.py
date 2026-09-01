#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time

import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx


def _ocean(shape, *, coriolis, directional):
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(shape[0], periodic=True),
            phx.discretization.UniformCellAxisSpec(shape[1], periodic=True),
            phx.discretization.UniformCellAxisSpec(shape[2], periodic=False),
        ),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray(((0.0, 0.0, -1.0), (1.0, 1.0, 0.0))))
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=("ocean",)
    ).prepare()
    diffusivity = jnp.asarray((1.0e-5, 1.0e-5, 1.0e-6)) if directional else 1.0e-5
    ocean = phx.applications.ocean.CartesianBoussinesqOceanPlan(
        phx.applications.ocean.OceanAxisConvention(),
        phx.applications.ocean.LinearSeawaterReference(),
        coriolis_parameter=1.0e-4 if coriolis else 0.0,
        temperature_diffusivity=diffusivity,
        salinity_diffusivity=diffusivity,
    ).prepare(discretization)
    velocity = tuple(jnp.zeros(layout.shape) for layout in discretization.face_layouts)
    z = grid.structured_axes[2].interval_centers
    temperature = 10.0 + 0.1 * jnp.broadcast_to(z.reshape((1, 1, z.size)), shape)
    coordinates = ocean.initial_state(
        velocity,
        temperature,
        jnp.full(shape, 35.0),
    )
    continuation = phx.applications.ocean.OceanBoussinesqContinuationState.initialize(
        coordinates
    )
    return ocean, continuation


def _measure(shape, repeats, *, coriolis, directional):
    ocean, continuation = _ocean(shape, coriolis=coriolis, directional=directional)
    rhs = eqx.filter_jit(ocean.dynamics)
    started = time.perf_counter()
    first = rhs(jnp.asarray(0.0), continuation.coordinates, None)
    jax.block_until_ready(first)
    rhs_compile = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(repeats):
        value = rhs(jnp.asarray(0.0), continuation.coordinates, None)
    jax.block_until_ready(value)
    rhs_seconds = (time.perf_counter() - started) / repeats

    method = phx.applications.ocean.OceanBoussinesqSSPRK33Method(ocean)
    step = eqx.filter_jit(method.step)
    dt = jnp.minimum(ocean.stable_step(continuation.coordinates), 0.01)
    started = time.perf_counter()
    first_step = step(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        continuation,
        dt,
        None,
    )
    jax.block_until_ready(first_step.accepted_state.coordinates)
    step_compile = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(repeats):
        step_value = step(
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0.0),
            continuation,
            dt,
            None,
        )
    jax.block_until_ready(step_value.accepted_state.coordinates)
    step_seconds = (time.perf_counter() - started) / repeats
    cells = shape[0] * shape[1] * shape[2]
    return {
        "shape": list(shape),
        "cells": cells,
        "repeats": repeats,
        "coriolis": coriolis,
        "directional_diffusion": directional,
        "rhs_compile_seconds": rhs_compile,
        "rhs_execution_seconds": rhs_seconds,
        "rhs_cell_evaluations_per_second": cells / rhs_seconds,
        "step_compile_seconds": step_compile,
        "step_execution_seconds": step_seconds,
        "step_cell_updates_per_second": cells / step_seconds,
        "successful": bool(step_value.successful),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", default="16,16,8")
    parser.add_argument("--repeats", type=int, default=5)
    arguments = parser.parse_args()
    shape = tuple(int(value) for value in arguments.shape.split(","))
    if len(shape) != 3 or any(value < 2 for value in shape):
        raise ValueError("Ocean benchmark shape must contain three counts >= 2.")
    if arguments.repeats <= 0:
        raise ValueError("Ocean benchmark repeats must be positive.")
    reports = [
        _measure(shape, arguments.repeats, coriolis=coriolis, directional=directional)
        for coriolis in (False, True)
        for directional in (False, True)
    ]
    print(json.dumps({"benchmarks": reports}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
