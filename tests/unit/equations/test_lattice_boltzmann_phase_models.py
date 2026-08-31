#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.discretization.lattice_boltzmann._boundary import (
    LatticeBoltzmannBoundaryPlan,
)
from phydrax.discretization.lattice_boltzmann._collision import BGKCollisionPlan
from phydrax.discretization.lattice_boltzmann._colour_gradient import (
    ColourGradientLBMMethod,
    ColourGradientLBMRuntimeParameters,
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
from phydrax.equations._lattice_boltzmann_colour_gradient import (
    ColourGradientLatticeBoltzmannProblem,
    compile_colour_gradient_lattice_boltzmann_problem,
)
from phydrax.equations._lattice_boltzmann_free_energy import (
    compile_free_energy_lattice_boltzmann_problem,
    FreeEnergyLatticeBoltzmannProblem,
)


def _discretization(count=24):
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(count, periodic=True),
            phx.discretization.UniformCellAxisSpec(count, periodic=True),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
    return LatticeBoltzmannPlan(grid, D2Q9()).prepare()


def _forced_method():
    return LatticeBoltzmannMethodPlan(BGKCollisionPlan(), forcing=GuoForcingPlan())


def test_colour_gradient_compiler_routes_both_populations_and_fails_atomically():
    discretization = _discretization()
    method = ColourGradientLBMMethod(_forced_method(), maximum_capillary_number=10.0)
    compiled = compile_colour_gradient_lattice_boltzmann_problem(
        ColourGradientLatticeBoltzmannProblem("binary", 2),
        discretization,
        method,
        LatticeBoltzmannBoundaryPlan(),
        time_step=0.01,
    )
    x = discretization.grid.points[:, 0].reshape(discretization.grid.shape)
    colour = jnp.tanh((x - 0.5) / 0.08)
    red = 0.5 * (1.0 + colour)
    blue = 1.0 - red
    parameters = ColourGradientLBMRuntimeParameters(0.01, 1.0e-4)
    state = compiled.initialize_state(red, blue, jnp.zeros((2,)), parameters)

    result = compiled.dynamics.step_detailed(0, 0.0, state, 0.01, parameters)
    assert result.candidate_state.red_populations.shape == discretization.population_shape
    assert (
        result.candidate_state.blue_populations.shape == discretization.population_shape
    )
    np.testing.assert_allclose(
        result.diagnostics.red_mass + result.diagnostics.blue_mass,
        result.diagnostics.total_mass,
        atol=1e-11,
    )
    assert result.diagnostics.recolouring.population_closure_defect <= 1e-11
    assert result.diagnostics.recolouring.momentum_closure_defect <= 1e-11

    rejected = compiled.dynamics.step_detailed(0, 0.0, state, 0.02, parameters)
    assert not bool(rejected.successful)
    np.testing.assert_array_equal(
        rejected.accepted_state.red_populations, state.red_populations
    )
    np.testing.assert_array_equal(
        rejected.accepted_state.blue_populations, state.blue_populations
    )


def test_free_energy_compiler_preserves_moments_and_has_nonincreasing_accepted_energy():
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
    initial = compiled.dynamics.scalar_diagnostics(0, 0.0, state, parameters)

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
    rejected = compiled.dynamics.step_detailed(0, 0.0, state, 0.01, invalid)
    assert not bool(rejected.successful)
    np.testing.assert_array_equal(
        rejected.accepted_state.hydrodynamic_populations,
        state.hydrodynamic_populations,
    )
    np.testing.assert_array_equal(
        rejected.accepted_state.phase_populations, state.phase_populations
    )
