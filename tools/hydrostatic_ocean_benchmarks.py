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


def _model(shape, *, mode, partial, nonlinear, closure):
    nx, ny, nz = shape
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(nx, periodic=True),
            phx.discretization.UniformCellAxisSpec(ny, periodic=True),
            phx.discretization.UniformCellAxisSpec(nz, periodic=False),
        ),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray(((0.0, 0.0, -100.0), (1.0, 1.0, 0.0))))
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=("hydrostatic",)
    ).prepare()
    geometry = phx.discretization.TensorZHydrostaticGridPlan(
        discretization,
        jnp.full((nx, ny), 100.0),
        vertical_coordinate="partial-z" if partial else "zstar",
    ).prepare()
    ocean = phx.applications.ocean.HydrostaticPrimitiveEquationPlan(
        geometry,
        eos=(
            phx.applications.ocean.NonlinearSeawaterPolynomialEOS() if nonlinear else None
        ),
        mixing=phx.applications.ocean.HydrostaticMixingPlan(closure),
        external_mode=mode,
        subcycle_policy=phx.applications.ocean.ExternalModeSubcyclePolicy.fixed(10),
    ).prepare()
    state = ocean.initialize_state(jnp.zeros((nx, ny)))
    continuation = phx.applications.ocean.HydrostaticContinuationState.initialize(
        ocean, state
    )
    method = phx.applications.ocean.HydrostaticIMEXMidpointMethod(ocean)
    return ocean, method, continuation


def _measure(shape, repeats, *, mode, partial, nonlinear, closure):
    ocean, method, state = _model(
        shape,
        mode=mode,
        partial=partial,
        nonlinear=nonlinear,
        closure=closure,
    )
    step = eqx.filter_jit(method.step)
    dt = jnp.asarray(0.01)
    started = time.perf_counter()
    first = step(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        state,
        dt,
        None,
    )
    jax.block_until_ready(first.accepted_state.state.eta)
    compile_seconds = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(repeats):
        result = step(
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0.0),
            state,
            dt,
            None,
        )
    jax.block_until_ready(result.accepted_state.state.eta)
    execution_seconds = (time.perf_counter() - started) / repeats
    cells = shape[0] * shape[1] * shape[2]
    return {
        "shape": list(shape),
        "cells": cells,
        "repeats": repeats,
        "external_mode": mode,
        "partial_cells": partial,
        "nonlinear_eos": nonlinear,
        "closure": closure,
        "compile_seconds": compile_seconds,
        "execution_seconds": execution_seconds,
        "cell_updates_per_second": cells / execution_seconds,
        "successful": bool(result.successful),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", default="8,8,4")
    parser.add_argument("--repeats", type=int, default=3)
    arguments = parser.parse_args()
    shape = tuple(int(value) for value in arguments.shape.split(","))
    if len(shape) != 3 or any(value < 3 for value in shape):
        raise ValueError("Hydrostatic benchmark shape needs three counts >= 3.")
    if arguments.repeats <= 0:
        raise ValueError("Hydrostatic benchmark repeats must be positive.")
    cases = (
        ("implicit", False, False, "prescribed"),
        ("implicit", True, True, "kpp"),
        ("split-explicit", False, False, "prescribed"),
        ("implicit", False, True, "redi-gm"),
    )
    reports = [
        _measure(
            shape,
            arguments.repeats,
            mode=mode,
            partial=partial,
            nonlinear=nonlinear,
            closure=closure,
        )
        for mode, partial, nonlinear, closure in cases
    ]
    print(json.dumps({"benchmarks": reports}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
