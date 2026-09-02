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


def _graph(shape, *, capillary, wave):
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(shape[0], periodic=True),
            phx.discretization.UniformCellAxisSpec(shape[1], periodic=True),
            phx.discretization.UniformCellAxisSpec(shape[2], periodic=False),
        ),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray(((0.0, 0.0, -1.0), (float(shape[0]), float(shape[1]), 0.0))))
    reference = phx.discretization.FiniteVolumePlan(
        grid, component_names=("hydrodynamics",)
    ).prepare()
    forcing = None
    if wave:
        provider = phx.equations.IncidentWavePlan(
            (phx.equations.WaveComponent(1.0e-4, 1.0),), 1.0
        )
        forcing = phx.applications.hydrodynamics.WaveForcingPlan(
            provider,
            jnp.zeros(shape).at[:2].set(0.5),
            jnp.zeros(shape).at[-2:].set(0.25),
        )
    surface = phx.applications.hydrodynamics.GraphSurfaceALEPlan(
        reference,
        jnp.full(shape[:2], -1.0),
        maximum_slope=0.8,
        maximum_iterations=100,
    )
    hydro = phx.applications.hydrodynamics.OnePhaseFreeSurfaceALEPlan(
        surface,
        surface_tension=0.072 if capillary else 0.0,
        wave=forcing,
        coupling_iterations=4,
        coupling_tolerance=1.0e-7,
    ).prepare()
    state = hydro.initial_state(jnp.zeros(shape[:2]))
    continuation = (
        phx.applications.hydrodynamics.FreeSurfaceALEContinuationState.initialize(state)
    )
    return phx.applications.hydrodynamics.OnePhaseFreeSurfaceALEMethod(
        hydro
    ), continuation


def _measure_graph(shape, repeats, *, capillary, wave):
    method, state = _graph(shape, capillary=capillary, wave=wave)
    dt = jnp.asarray(1.0e-4 if (capillary or wave) else 1.0e-3)
    step = eqx.filter_jit(method.step)
    started = time.perf_counter()
    first = step(jnp.asarray(0, dtype=jnp.int32), jnp.asarray(0.0), state, dt, None)
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
    execution = (time.perf_counter() - started) / repeats
    cells = shape[0] * shape[1] * shape[2]
    return {
        "product": "graph-ale",
        "capillary": capillary,
        "wave": wave,
        "shape": list(shape),
        "compile_seconds": compile_seconds,
        "execution_seconds": execution,
        "cell_updates_per_second": cells / execution,
        "successful": bool(result.successful),
    }


def _measure_two_phase(repeats):
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(8, periodic=True),
            phx.discretization.UniformCellAxisSpec(8, periodic=True),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=("two-phase",)
    ).prepare()
    two_phase = phx.applications.two_phase_flow.IncompressibleTwoPhaseVOFPlan(
        discretization,
        phx.applications.two_phase_flow.TwoPhaseMaterialPlan(
            liquid_density=1000.0,
            gas_density=10.0,
            surface_tension=0.072,
        ),
    ).prepare()
    alpha = jnp.where(jnp.indices((8, 8))[0] < 4, 1.0, 0.0)
    method = phx.applications.two_phase_flow.IncompressibleTwoPhaseVOFMethod(two_phase)
    state = method.initial_continuation(two_phase.initial_state(alpha))
    step = eqx.filter_jit(method.step)
    started = time.perf_counter()
    first = step(
        jnp.asarray(0, dtype=jnp.int32), jnp.asarray(0.0), state, jnp.asarray(0.001), None
    )
    jax.block_until_ready(first.accepted_state.state.liquid_content)
    compile_seconds = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(repeats):
        result = step(
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0.0),
            state,
            jnp.asarray(0.001),
            None,
        )
    jax.block_until_ready(result.accepted_state.state.liquid_content)
    execution = (time.perf_counter() - started) / repeats
    return {
        "product": "two-phase-vof",
        "shape": [8, 8],
        "compile_seconds": compile_seconds,
        "execution_seconds": execution,
        "cell_updates_per_second": 64 / execution,
        "successful": bool(result.successful),
    }


def _measure_passive_tracer(repeats):
    shape = (64, 64)
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(count, periodic=True)
            for count in shape
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
    discretization = phx.discretization.FiniteVolumePlan(grid).prepare()
    mac = phx.discretization.MACOperatorPlan(discretization).prepare()
    space = grid.field_space(
        "passive-tracer",
        entity_layout=discretization.cell_layout,
        dtype=mac.pressure_space.dtype,
        representation="point_value",
    )
    transport = phx.discretization.MACPassiveTracerMacCormackPlan(
        mac,
        space,
    ).prepare()
    values = jnp.exp(
        -160.0
        * jnp.sum((discretization.cell_centers - jnp.asarray((0.35, 0.4))) ** 2, axis=-1)
    )
    velocity = (
        jnp.full(discretization.face_layouts[0].shape, 0.25),
        jnp.full(discretization.face_layouts[1].shape, -0.1),
    )
    dt = jnp.asarray(0.001)
    advance = eqx.filter_jit(transport.advance)
    started = time.perf_counter()
    first = advance(values, velocity, dt)
    jax.block_until_ready(first.values)
    compile_seconds = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(repeats):
        result = advance(values, velocity, dt)
    jax.block_until_ready(result.values)
    execution = (time.perf_counter() - started) / repeats

    def objective(tracer):
        return jnp.sum(transport.advance(tracer, velocity, dt).values ** 2)

    gradient = eqx.filter_jit(jax.grad(objective))
    started = time.perf_counter()
    first_gradient = gradient(values)
    jax.block_until_ready(first_gradient)
    gradient_compile_seconds = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(repeats):
        gradient_result = gradient(values)
    jax.block_until_ready(gradient_result)
    gradient_execution = (time.perf_counter() - started) / repeats
    cells = shape[0] * shape[1]
    return {
        "product": "passive-tracer-maccormack",
        "shape": list(shape),
        "compile_seconds": compile_seconds,
        "execution_seconds": execution,
        "cell_updates_per_second": cells / execution,
        "gradient_compile_seconds": gradient_compile_seconds,
        "gradient_execution_seconds": gradient_execution,
        "integral_defect": float(result.integral_defect),
        "limiter_cells": int(result.limiter_active_count),
        "successful": bool(result.success),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", default="4,4,3")
    parser.add_argument("--repeats", type=int, default=2)
    arguments = parser.parse_args()
    shape = tuple(int(value) for value in arguments.shape.split(","))
    if len(shape) != 3 or any(value < 3 for value in shape):
        raise ValueError("Advanced hydrodynamic benchmark shape needs counts >= 3.")
    if arguments.repeats <= 0:
        raise ValueError("Advanced hydrodynamic benchmark repeats must be positive.")
    reports = [
        _measure_graph(shape, arguments.repeats, capillary=False, wave=False),
        _measure_graph(shape, arguments.repeats, capillary=True, wave=False),
        _measure_graph(shape, arguments.repeats, capillary=True, wave=True),
        _measure_two_phase(arguments.repeats),
        _measure_passive_tracer(arguments.repeats),
    ]
    print(json.dumps({"benchmarks": reports}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
