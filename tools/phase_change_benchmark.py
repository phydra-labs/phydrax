#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp

import phydrax as phx


def _measure(function, argument, repetitions):
    compiled = jax.jit(function)
    start = time.perf_counter()
    first = compiled(argument)
    jax.block_until_ready(first)
    compile_seconds = time.perf_counter() - start
    start = time.perf_counter()
    value = None
    for _ in range(repetitions):
        value = compiled(argument)
    jax.block_until_ready(value)
    elapsed = time.perf_counter() - start
    return {
        "compile_seconds": compile_seconds,
        "steady_seconds_per_call": elapsed / repetitions,
        "repetitions": repetitions,
    }


def _enthalpy(size, repetitions):
    material = phx.equations.SolidLiquidEnthalpyPlan(
        1000.0, 273.15, 300.0, 302.0, 2000.0, 2200.0, 2.0e5, 2.0, 0.5, 1.0e-6
    )
    enthalpy = material.enthalpy_from_temperature(jnp.linspace(280.0, 320.0, size))
    return _measure(
        lambda value: material.evaluate(value).temperature,
        enthalpy,
        repetitions,
    )


def _hem(size, repetitions):
    material = phx.equations.HomogeneousEquilibriumCavitationMaterial(
        1.0e5,
        1.0,
        1000.0,
        300.0,
        1400.0,
        mixture_model="wallis",
        pressure_floor=-1.0e6,
    )
    density = jnp.geomspace(1.0, 1000.0, size)
    return _measure(
        lambda value: material.evaluate(value).pressure,
        density,
        repetitions,
    )


def _vof(size, repetitions):
    eos = phx.equations.TwoMaterialEOSClosure(
        phx.equations.StiffenedGasMaterial(4.4, 2.0e5, 1800.0),
        phx.equations.StiffenedGasMaterial(1.33, 0.0, 1400.0, reference_energy=2.0e6),
    )
    system = phx.equations.TwoMaterialVOFSystem(1, eos=eos)
    alpha = jnp.linspace(0.05, 0.95, size)
    state = system.primitive_to_conserved(
        jnp.stack(
            (
                jnp.full_like(alpha, 900.0),
                jnp.full_like(alpha, 2.0),
                jnp.zeros_like(alpha),
                jnp.full_like(alpha, 8.0e4),
                alpha,
            ),
            axis=-1,
        )
    )
    transfer = phx.equations.TwoMaterialVOFPhaseChangePlan(
        system,
        phx.equations.SchnerrSauerCavitationPlan(1.0e5, 1.0e8, 1.0e-5, 1.0, 1.0, 2.0e6),
    )
    return _measure(
        lambda value: transfer.differential_source(value).state_rate,
        state,
        repetitions,
    )


def _mac(size, repetitions):
    count = max(4, int(size**0.5))
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(count, periodic=True),
            phx.discretization.UniformCellAxisSpec(count, periodic=True),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    finite_volume = phx.discretization.FiniteVolumePlan(grid).prepare()
    operators = phx.discretization.MACOperatorPlan(finite_volume).prepare()
    momentum = phx.discretization.MACMomentumPlan(operators).prepare()
    projection = phx.solver.MACPressureProjectionPlan(
        operators, solve_method="transform", tolerance=1.0e-9
    )
    material = phx.equations.SolidLiquidEnthalpyPlan(
        1000.0,
        273.15,
        300.0,
        302.0,
        2000.0,
        2200.0,
        2.0e5,
        2.0,
        0.5,
        0.01,
        thermal_expansion=2.0e-4,
        mushy_resistance_coefficient=1.0e3,
    )
    transport = phx.discretization.MACEnthalpyTransportPlan(
        operators, phx.discretization.MACThermalBoundarySet(operators)
    ).prepare()
    dynamics = phx.equations.compile_mac_enthalpy_porosity(
        phx.equations.IncompressibleFlowProblem(2, 0.01),
        phx.equations.MACEnthalpyPorosityProblem(material, jnp.asarray([0.0, -9.81])),
        momentum,
        projection,
        transport,
    )
    velocity = tuple(
        jnp.zeros(layout.shape, dtype=operators.pressure_space.dtype)
        for layout in finite_volume.face_layouts
    )
    temperature = 301.0 + 0.5 * jnp.sin(2.0 * jnp.pi * finite_volume.cell_centers[..., 0])
    state = dynamics.pack_state(velocity, material.enthalpy_from_temperature(temperature))
    return _measure(
        lambda value: dynamics(jnp.asarray(0.0), value, None),
        state,
        repetitions,
    )


_CASES = {
    "enthalpy": _enthalpy,
    "hem": _hem,
    "mac": _mac,
    "vof": _vof,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=(*_CASES, "all"), default="all")
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--repetitions", type=int, default=20)
    arguments = parser.parse_args()
    names = tuple(_CASES) if arguments.case == "all" else (arguments.case,)
    results = {
        name: _CASES[name](arguments.size, arguments.repetitions) for name in names
    }
    print(json.dumps({"size": arguments.size, "cases": results}, allow_nan=False))


if __name__ == "__main__":
    main()
