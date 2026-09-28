#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization.lattice_boltzmann._boundary import (
    LatticeBoltzmannBoundaryPlan,
)
from phydrax.discretization.lattice_boltzmann._collision import BGKCollisionPlan
from phydrax.discretization.lattice_boltzmann._color_gradient import (
    ColorGradientLBMMethod,
    ColorGradientLBMRuntimeParameters,
)
from phydrax.discretization.lattice_boltzmann._discretization import (
    LatticeBoltzmannPlan,
)
from phydrax.discretization.lattice_boltzmann._forcing import GuoForcingPlan
from phydrax.discretization.lattice_boltzmann._free_energy import (
    FreeEnergyLBMMethod,
    FreeEnergyLBMRuntimeParameters,
)
from phydrax.discretization.lattice_boltzmann._lattice import D2Q9
from phydrax.discretization.lattice_boltzmann._method import LatticeBoltzmannMethodPlan
from phydrax.equations._lattice_boltzmann_color_gradient import (
    ColorGradientLatticeBoltzmannProblem,
    compile_color_gradient_lattice_boltzmann_problem,
)
from phydrax.equations._lattice_boltzmann_free_energy import (
    compile_free_energy_lattice_boltzmann_problem,
    FreeEnergyLatticeBoltzmannProblem,
)
from phydrax.interfacial_transport import InterfaceTensionMatrix


def _discretization(count: Any = 24) -> Any:
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(count, periodic=True),
            phx.discretization.UniformCellAxisSpec(count, periodic=True),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
    return LatticeBoltzmannPlan(grid, D2Q9()).prepare()


def _forced_method() -> Any:
    return LatticeBoltzmannMethodPlan(BGKCollisionPlan(), forcing=GuoForcingPlan())


def test_lattice_boltzmann_phase_models_scenario_1() -> None:
    discretization = _discretization()
    method = ColorGradientLBMMethod(
        _forced_method(), ("red", "blue"), maximum_capillary_number=10.0
    )
    compiled = compile_color_gradient_lattice_boltzmann_problem(
        ColorGradientLatticeBoltzmannProblem("binary", 2),
        discretization,
        method,
        LatticeBoltzmannBoundaryPlan(),
        time_step=0.01,
    )
    x = discretization.grid.points[:, 0].reshape(discretization.grid.shape)
    color = jnp.tanh((x - 0.5) / 0.08)
    red = 0.5 * (1.0 + color)
    blue = 1.0 - red
    tension = InterfaceTensionMatrix(
        ("red", "blue"), np.asarray([[0.0, 1.0e-4], [1.0e-4, 0.0]])
    )
    parameters = ColorGradientLBMRuntimeParameters(0.01, tension)
    state = compiled.initialize_state(jnp.stack((red, blue)), jnp.zeros((2,)), parameters)

    # ty: ignore[invalid-argument-type]
    result = compiled.dynamics.step_detailed(0, 0.0, state, 0.01, parameters)
    assert result.candidate_state.color_populations.shape == (
        2,
        *discretization.population_shape,
    )
    np.testing.assert_allclose(
        jnp.sum(result.diagnostics.component_masses),
        result.diagnostics.total_mass,
        atol=1e-11,
    )
    assert result.diagnostics.recoloring.population_closure_defect <= 1e-11
    assert result.diagnostics.recoloring.momentum_closure_defect <= 1e-11

    # ty: ignore[invalid-argument-type]
    rejected = compiled.dynamics.step_detailed(0, 0.0, state, 0.02, parameters)
    assert not bool(rejected.successful)
    np.testing.assert_array_equal(
        rejected.accepted_state.color_populations, state.color_populations
    )
    np.testing.assert_array_equal(
        rejected.accepted_state.near_contact_work, state.near_contact_work
    )
    discretization = _discretization()
    method = FreeEnergyLBMMethod(
        _forced_method(),
        phx.equations.BinaryPhaseThermodynamicClosure(),
        phx.equations.ThermodynamicForceRepresentation.CHEMICAL_POTENTIAL_GRADIENT,
        maximum_capillary_number=10.0,
        relative_energy_tolerance=1.0e-6,
    )
    compiled = compile_free_energy_lattice_boltzmann_problem(
        FreeEnergyLatticeBoltzmannProblem("cahn-hilliard", 2),
        discretization,
        method,
        LatticeBoltzmannBoundaryPlan(),
        time_step=0.01,
    )
    x = discretization.grid.points[:, 0].reshape(discretization.grid.shape)
    phase = jnp.tanh((x - 0.5) / 0.08)
    parameters = FreeEnergyLBMRuntimeParameters(
        0.01,
        0.08,
        phx.equations.BinaryThermodynamicParameters(0.02, 0.02),
    )
    state = compiled.initialize_state(1.0, phase, jnp.zeros((2,)), parameters)
    # ty: ignore[invalid-argument-type]
    initial = compiled.dynamics.scalar_diagnostics(0, 0.0, state, parameters)

    # ty: ignore[invalid-argument-type]
    result = compiled.dynamics.step_detailed(0, 0.0, state, 0.01, parameters)
    assert (
        result.candidate_state.hydrodynamic_populations.shape
        == discretization.population_shape
    )
    assert (
        result.candidate_state.phase_populations.shape == discretization.population_shape
    )
    assert result.diagnostics.mixture_mass_defect <= method.conservation_tolerance
    assert result.diagnostics.phase_mass_defect <= method.conservation_tolerance
    assert (
        result.diagnostics.phase_equilibrium_mass_defect <= method.conservation_tolerance
    )
    assert (
        result.diagnostics.phase_equilibrium_flux_defect <= method.conservation_tolerance
    )
    assert result.diagnostics.ledger.total_energy <= (
        initial.ledger.total_energy
        + method.relative_energy_tolerance * jnp.maximum(initial.ledger.total_energy, 1.0)
    )

    invalid = FreeEnergyLBMRuntimeParameters(
        0.01,
        0.0,
        phx.equations.BinaryThermodynamicParameters(0.02, 0.02),
    )
    # ty: ignore[invalid-argument-type]
    rejected = compiled.dynamics.step_detailed(0, 0.0, state, 0.01, invalid)
    assert not bool(rejected.successful)
    np.testing.assert_array_equal(
        rejected.accepted_state.hydrodynamic_populations,
        state.hydrodynamic_populations,
    )
    np.testing.assert_array_equal(
        rejected.accepted_state.phase_populations, state.phase_populations
    )
    discretization = _discretization()
    method = FreeEnergyLBMMethod(
        _forced_method(),
        phx.equations.BinaryPhaseThermodynamicClosure(),
        phx.equations.ThermodynamicForceRepresentation.STRESS_DIVERGENCE,
        maximum_cells=int(np.prod(discretization.grid.shape)) - 1,
    )

    with pytest.raises(ValueError, match="maximum_cells"):
        compile_free_energy_lattice_boltzmann_problem(
            FreeEnergyLatticeBoltzmannProblem("cahn-hilliard", 2),
            discretization,
            method,
            LatticeBoltzmannBoundaryPlan(),
            time_step=0.01,
        )
