#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json

import jax.numpy as jnp

import phydrax as phx


def _ocean(*, shape, coriolis=0.0, temperature_flux=None):
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
    return phx.applications.ocean.CartesianBoussinesqOceanPlan(
        phx.applications.ocean.OceanAxisConvention(),
        phx.applications.ocean.LinearSeawaterReference(),
        coriolis_parameter=coriolis,
        temperature_surface_flux=temperature_flux,
        temperature_diffusivity=jnp.asarray((1.0e-5, 1.0e-5, 1.0e-6)),
        salinity_diffusivity=jnp.asarray((1.0e-5, 1.0e-5, 1.0e-6)),
    ).prepare(discretization)


def _state(ocean, *, u=0.0, temperature=None):
    discretization = ocean.operators.discretization
    velocity = (
        jnp.full(discretization.face_layouts[0].shape, u),
        jnp.zeros(discretization.face_layouts[1].shape),
        jnp.zeros(discretization.face_layouts[2].shape),
    )
    reference = ocean.plan.reference
    temperature_ = (
        jnp.full(discretization.cell_shape, reference.reference_temperature)
        if temperature is None
        else temperature
    )
    salinity = jnp.full(discretization.cell_shape, reference.reference_salinity)
    return ocean.initial_state(velocity, temperature_, salinity)


def _rest(shape):
    ocean = _ocean(shape=shape)
    coordinates = _state(ocean)
    stage = ocean.dynamics.stage(0.0, coordinates)
    diagnostics = ocean.dynamics.diagnostics(0.0, coordinates)
    maximum_rate = float(jnp.max(jnp.abs(ocean.dynamics(0.0, coordinates, None))))
    return {
        "case": "rest",
        "shape": list(shape),
        "maximum_rate": maximum_rate,
        "divergence_norm": float(diagnostics.divergence_norm),
        "exchange_defect": float(stage.buoyancy.exchange_defect),
        "passed": bool(stage.success) and maximum_rate <= 1.0e-11,
    }


def _inertial(shape, steps, dt):
    ocean = _ocean(shape=shape, coriolis=0.5)
    continuation = phx.applications.ocean.OceanBoussinesqContinuationState.initialize(
        _state(ocean, u=1.0)
    )
    method = phx.applications.ocean.OceanBoussinesqSSPRK33Method(ocean)
    time = jnp.asarray(0.0)
    accepted = True
    for step in range(steps):
        result = method.step(
            jnp.asarray(step, dtype=jnp.int32),
            time,
            continuation,
            jnp.asarray(dt),
            None,
        )
        continuation = result.accepted_state
        time = time + dt
        accepted = accepted and bool(result.successful)
    view = ocean.state_view(continuation.coordinates)
    expected_u = jnp.cos(0.5 * time)
    expected_v = -jnp.sin(0.5 * time)
    error = jnp.maximum(
        jnp.max(jnp.abs(view.velocity[0] - expected_u)),
        jnp.max(jnp.abs(view.velocity[1] - expected_v)),
    )
    return {
        "case": "inertial",
        "shape": list(shape),
        "steps": steps,
        "time": float(time),
        "maximum_error": float(error),
        "coriolis_work": float(continuation.coriolis_work),
        "passed": accepted and float(error) <= 5.0e-7,
    }


def _stratified(shape):
    ocean = _ocean(shape=shape, coriolis=1.0e-4)
    z = ocean.operators.discretization.grid.structured_axes[2].interval_centers
    temperature = 10.0 + jnp.broadcast_to(z.reshape((1, 1, z.size)), shape)
    coordinates = _state(ocean, temperature=temperature)
    restriction = ocean.dynamics.step_restriction(coordinates)
    stage = ocean.dynamics.stage(0.0, coordinates)
    return {
        "case": "stratified",
        "shape": list(shape),
        "stratification_step": float(restriction.stratification),
        "selected_step": float(restriction.selected),
        "exchange_defect": float(stage.buoyancy.exchange_defect),
        "passed": bool(stage.success) and bool(jnp.isfinite(restriction.stratification)),
    }


def _surface_flux(shape, dt):
    flux = phx.discretization.MACScalarBoundaryCondition("flux", 1.0e-5)
    ocean = _ocean(shape=shape, temperature_flux=flux)
    coordinates = _state(ocean)
    continuation = phx.applications.ocean.OceanBoussinesqContinuationState.initialize(
        coordinates
    )
    result = phx.applications.ocean.OceanBoussinesqSSPRK33Method(ocean).step(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        continuation,
        jnp.asarray(dt),
        None,
    )
    before = ocean.state_view(coordinates).temperature
    after = ocean.state_view(result.accepted_state.coordinates).temperature
    content = jnp.sum(ocean.operators.discretization.cell_volumes * (after - before))
    defect = content - result.accepted_state.temperature_boundary_content
    return {
        "case": "surface-flux",
        "shape": list(shape),
        "content_change": float(content),
        "ledger_defect": float(defect),
        "passed": bool(result.successful) and abs(float(defect)) <= 1.0e-11,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=("rest", "inertial", "stratified", "surface-flux"),
        default="rest",
    )
    parser.add_argument("--shape", default="8,8,8")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--dt", type=float, default=0.01)
    arguments = parser.parse_args()
    shape = tuple(int(value) for value in arguments.shape.split(","))
    if len(shape) != 3 or any(value < 2 for value in shape):
        raise ValueError("Ocean qualification shape must contain three counts >= 2.")
    if arguments.steps <= 0 or arguments.dt <= 0.0:
        raise ValueError("Ocean qualification steps and dt must be positive.")
    if arguments.case == "rest":
        report = _rest(shape)
    elif arguments.case == "inertial":
        report = _inertial(shape, arguments.steps, arguments.dt)
    elif arguments.case == "stratified":
        report = _stratified(shape)
    else:
        report = _surface_flux(shape, arguments.dt)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
