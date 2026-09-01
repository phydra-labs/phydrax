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


def _model(shape, *, wave):
    nx, ny, nz = shape
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(nx, periodic=True),
            phx.discretization.UniformCellAxisSpec(ny, periodic=True),
            phx.discretization.UniformCellAxisSpec(nz, periodic=False),
        ),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray(((0.0, 0.0, -1.0), (float(nx), float(ny), 0.0))))
    reference = phx.discretization.FiniteVolumePlan(
        grid, component_names=("hydrodynamics",)
    ).prepare()
    surface = phx.applications.hydrodynamics.GraphSurfaceALEPlan(
        reference,
        jnp.full((nx, ny), -1.0),
        maximum_slope=0.5,
        maximum_iterations=100,
    )
    hydro = phx.applications.hydrodynamics.OnePhaseFreeSurfaceALEPlan(
        surface,
        coupling_iterations=4,
        coupling_tolerance=1.0e-7,
    ).prepare()
    eta = jnp.zeros((nx, ny))
    if wave:
        eta = jnp.broadcast_to(
            1.0e-4 * jnp.sin(2.0 * jnp.pi * jnp.arange(nx)[:, None] / nx),
            (nx, ny),
        )
    state = hydro.initial_state(eta)
    continuation = (
        phx.applications.hydrodynamics.FreeSurfaceALEContinuationState.initialize(state)
    )
    method = phx.applications.hydrodynamics.OnePhaseFreeSurfaceALEMethod(hydro)
    return hydro, method, continuation


def _measure(shape, repeats, *, wave):
    hydro, method, state = _model(shape, wave=wave)
    geometry = eqx.filter_jit(hydro.surface.geometry)
    zero_rate = jnp.zeros_like(state.state.eta)
    started = time.perf_counter()
    first_geometry = geometry(jnp.asarray(0.0), state.state.eta, zero_rate)
    jax.block_until_ready(first_geometry.cell_volumes)
    geometry_compile = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(repeats):
        geometry_value = geometry(jnp.asarray(0.0), state.state.eta, zero_rate)
    jax.block_until_ready(geometry_value.cell_volumes)
    geometry_seconds = (time.perf_counter() - started) / repeats

    step = eqx.filter_jit(method.step)
    started = time.perf_counter()
    first = step(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        state,
        jnp.asarray(0.002),
        None,
    )
    jax.block_until_ready(first.accepted_state.state.eta)
    step_compile = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(repeats):
        result = step(
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0.0),
            state,
            jnp.asarray(0.002),
            None,
        )
    jax.block_until_ready(result.accepted_state.state.eta)
    step_seconds = (time.perf_counter() - started) / repeats
    cells = shape[0] * shape[1] * shape[2]
    return {
        "shape": list(shape),
        "cells": cells,
        "wave": wave,
        "repeats": repeats,
        "geometry_compile_seconds": geometry_compile,
        "geometry_execution_seconds": geometry_seconds,
        "geometry_cells_per_second": cells / geometry_seconds,
        "step_compile_seconds": step_compile,
        "step_execution_seconds": step_seconds,
        "step_cells_per_second": cells / step_seconds,
        "successful": bool(result.successful),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", default="6,6,3")
    parser.add_argument("--repeats", type=int, default=2)
    arguments = parser.parse_args()
    shape = tuple(int(value) for value in arguments.shape.split(","))
    if len(shape) != 3 or any(value < 3 for value in shape):
        raise ValueError("Hydrodynamic benchmark shape needs three counts >= 3.")
    if arguments.repeats <= 0:
        raise ValueError("Hydrodynamic benchmark repeats must be positive.")
    reports = [_measure(shape, arguments.repeats, wave=wave) for wave in (False, True)]
    print(json.dumps({"benchmarks": reports}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
