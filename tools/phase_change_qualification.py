#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import math

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _number(value):
    return float(np.asarray(value))


def _solid_material():
    return phx.equations.SolidLiquidEnthalpyPlan(
        1000.0,
        273.15,
        300.0,
        300.0,
        2000.0,
        2200.0,
        2.0e5,
        2.0,
        0.5,
        1.0e-6,
        mushy_resistance_coefficient=1.0e5,
    )


def _stefan():
    material = _solid_material()
    count = 64
    spacing = 1.0 / count
    cold = material.enthalpy_from_temperature(
        jnp.asarray(300.0), liquid_fraction=jnp.asarray(0.0)
    )
    hot = material.enthalpy_from_temperature(jnp.asarray(320.0))
    initial = jnp.full((count,), cold)
    diffusivity = max(
        material.solid_conductivity
        / (material.reference_density * material.solid_heat_capacity),
        material.liquid_conductivity
        / (material.reference_density * material.liquid_heat_capacity),
    )
    step = 0.2 * spacing**2 / diffusivity
    steps = 200

    def advance(_, enthalpy):
        state = material.evaluate(enthalpy)
        conductivity = state.conductivity
        face_k = (
            2.0
            * conductivity[:-1]
            * conductivity[1:]
            / (conductivity[:-1] + conductivity[1:])
        )
        internal = face_k * (state.temperature[1:] - state.temperature[:-1]) / spacing
        left_temperature = material.evaluate(hot).temperature
        left = 2.0 * conductivity[0] * (left_temperature - state.temperature[0]) / spacing
        rate = jnp.zeros_like(enthalpy)
        rate = rate.at[0].add(left / spacing)
        rate = rate.at[:-1].add(internal / spacing)
        rate = rate.at[1:].add(-internal / spacing)
        return enthalpy + step * rate

    final = jax.lax.fori_loop(0, steps, advance, initial)
    state = material.evaluate(final)
    melt_length = jnp.sum(state.liquid_fraction) * spacing
    energy_gain = jnp.sum(final - initial) * spacing
    stefan_number = (
        material.liquid_heat_capacity
        * (320.0 - material.liquidus_temperature)
        / material.latent_heat
    )
    lower, upper = 0.0, 2.0
    for _ in range(80):
        middle = 0.5 * (lower + upper)
        value = math.sqrt(math.pi) * middle * math.exp(middle**2) * math.erf(middle)
        if value < stefan_number:
            lower = middle
        else:
            upper = middle
    similarity = 0.5 * (lower + upper)
    liquid_diffusivity = material.liquid_conductivity / (
        material.reference_density * material.liquid_heat_capacity
    )
    exact_front = 2.0 * similarity * math.sqrt(liquid_diffusivity * step * steps)
    front_error = jnp.abs(melt_length - exact_front)
    passed = (
        jnp.all(state.successful)
        & (melt_length > 0.0)
        & (melt_length < 1.0)
        & (energy_gain > 0.0)
        & (front_error <= 4.0 * spacing)
    )
    return {
        "passed": bool(np.asarray(passed)),
        "melt_length": _number(melt_length),
        "energy_gain": _number(energy_gain),
        "exact_front": exact_front,
        "front_error": _number(front_error),
    }


def _hem_cavitation():
    material = phx.equations.HomogeneousEquilibriumCavitationMaterial(
        1.0e5,
        1.0,
        1000.0,
        300.0,
        1400.0,
        mixture_model="wallis",
        pressure_floor=-1.0e6,
    )
    density = jnp.geomspace(1.0, 1000.0, 256)
    state = material.evaluate(density)
    recovered = material.density_from_pressure(state.pressure)
    inverse_error = jnp.max(jnp.abs(recovered - density) / density)
    passed = (
        jnp.all(state.successful)
        & (inverse_error < 2.0e-4)
        & (jnp.min(state.sound_speed) > 0.0)
    )
    return {
        "passed": bool(np.asarray(passed)),
        "inverse_relative_error": _number(inverse_error),
        "minimum_sound_speed": _number(jnp.min(state.sound_speed)),
    }


def _vof_transfer(thermal):
    eos = phx.equations.TwoMaterialEOSClosure(
        phx.equations.StiffenedGasMaterial(4.4, 2.0e5, 1800.0),
        phx.equations.StiffenedGasMaterial(1.33, 0.0, 1400.0, reference_energy=2.0e6),
    )
    system = phx.equations.TwoMaterialVOFSystem(1, eos=eos)
    primitive = jnp.asarray([[900.0, 2.0, 0.0, 8.0e4, 0.6]])
    state = system.primitive_to_conserved(primitive)
    law = (
        phx.equations.InterfaceHeatResistancePhaseChangePlan(300.0, 100.0, 2.0e6)
        if thermal
        else phx.equations.SchnerrSauerCavitationPlan(
            1.0e5, 1.0e8, 1.0e-5, 1.0, 1.0, 2.0e6
        )
    )
    plan = phx.equations.TwoMaterialVOFPhaseChangePlan(system, law)
    result = plan.step(
        state,
        1.0e-6,
        interface_area_density=jnp.asarray([2.0]),
    )
    mass_defect = jnp.max(jnp.abs(result.transfer.mass_defect))
    energy_defect = jnp.max(
        jnp.abs(
            result.state[..., system.layout.energy_index]
            - state[..., system.layout.energy_index]
        )
    )
    passed = (
        jnp.all(result.accepted)
        & (mass_defect == 0.0)
        & (energy_defect == 0.0)
        & jnp.all(system.admissible(result.state))
    )
    return {
        "passed": bool(np.asarray(passed)),
        "mass_defect": _number(mass_defect),
        "energy_defect": _number(energy_defect),
        "transfer_rate": _number(result.transfer.limited_mass_rate[0]),
    }


def _melting_cavity():
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(4, periodic=True),
            phx.discretization.UniformCellAxisSpec(4, periodic=True),
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
    thermal = phx.discretization.MACEnthalpyTransportPlan(
        operators, phx.discretization.MACThermalBoundarySet(operators)
    ).prepare()
    dynamics = phx.equations.compile_mac_enthalpy_porosity(
        phx.equations.IncompressibleFlowProblem(2, 0.01),
        phx.equations.MACEnthalpyPorosityProblem(material, jnp.asarray([0.0, -9.81])),
        momentum,
        projection,
        thermal,
    )
    velocity = tuple(
        jnp.zeros(layout.shape, dtype=operators.pressure_space.dtype)
        for layout in finite_volume.face_layouts
    )
    temperature = 299.0 + 4.0 * finite_volume.cell_centers[..., 0]
    enthalpy = material.enthalpy_from_temperature(temperature)
    state = dynamics.pack_state(velocity, enthalpy)
    diagnostics = dynamics.diagnostics(0.0, state)
    passed = diagnostics.successful & diagnostics.projection_converged
    return {
        "passed": bool(np.asarray(passed)),
        "melt_fraction": _number(jnp.mean(material.evaluate(enthalpy).liquid_fraction)),
        "enthalpy_balance_defect": _number(diagnostics.enthalpy.balance_defect),
        "divergence_norm": _number(diagnostics.divergence_norm),
    }


_CASES = {
    "stefan": _stefan,
    "melting-cavity": _melting_cavity,
    "hem-cavitation": _hem_cavitation,
    "kinetic-cavitation": lambda: _vof_transfer(False),
    "thermal-vof": lambda: _vof_transfer(True),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=(*_CASES, "all"), default="all")
    arguments = parser.parse_args()
    names = tuple(_CASES) if arguments.case == "all" else (arguments.case,)
    results = {name: _CASES[name]() for name in names}
    passed = all(result["passed"] for result in results.values())
    print(json.dumps({"passed": passed, "cases": results}, allow_nan=False))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
