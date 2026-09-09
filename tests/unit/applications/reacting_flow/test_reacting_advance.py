#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.discretization.finite_volume._dynamics import PreparedFiniteVolumeDynamics
from phydrax.discretization.finite_volume._positivity import FluxPositivityPlan
from phydrax.discretization.finite_volume._riemann import RusanovFluxPlan
from phydrax.equations._chemical_mechanism import (
    ChemicalMechanismIR,
    ChemicalReactionSpec,
)
from phydrax.equations._chemical_rates import ArrheniusRatePlan
from phydrax.equations._chemical_species import ChemicalPhaseKind, ChemicalSpeciesSchema
from phydrax.equations._chemical_thermodynamics import (
    PolynomialSpeciesThermodynamicsPlan,
)
from phydrax.equations._gas_dynamics import (
    HomogeneousMixtureCompressibleNavierStokesSystem,
    HomogeneousMixtureEulerSystem,
)
from phydrax.equations._homogeneous_thermodynamics import (
    HomogeneousHelmholtzPlan,
    IdealGasReferenceHelmholtzTerm,
    ZeroResidualHelmholtzTerm,
)
from phydrax.equations._transport_closures import ConstantTransport
from phydrax.solver._finite_volume_runtime import PreparedFiniteVolumeRuntime


def _problem(rate=1.0, *, viscous=False):
    schema = ChemicalSpeciesSchema.from_unique_species(
        ("A", "B"),
        (ChemicalPhaseKind.GAS, ChemicalPhaseKind.GAS),
        jnp.asarray((0.01, 0.01)),
        ("E",),
        jnp.asarray(((1, 1),), dtype=jnp.int32),
        jnp.asarray((0, 0), dtype=jnp.int32),
        gas_standard_pressure=1.0e5,
        provenance="thermochemistry-process-test",
    )
    species_thermodynamics = PolynomialSpeciesThermodynamicsPlan(
        schema,
        jnp.asarray((20.0, 20.0)),
        jnp.asarray((0.0, -5.0e4)),
        reference_temperature=300.0,
        minimum_temperature=200.0,
        maximum_temperature=3000.0,
    )
    thermodynamics = HomogeneousHelmholtzPlan(
        IdealGasReferenceHelmholtzTerm(schema, species_thermodynamics),
        ZeroResidualHelmholtzTerm(schema),
    )
    system = (
        HomogeneousMixtureCompressibleNavierStokesSystem(
            thermodynamics, ConstantTransport(1.0e-5, 1.0e-2), 1
        )
        if viscous
        else HomogeneousMixtureEulerSystem(
            thermodynamics, 1, maximum_thermal_iterations=48
        )
    )
    mechanism = ChemicalMechanismIR(
        "A-to-B",
        schema,
        species_thermodynamics,
        (
            ChemicalReactionSpec(
                "A->B",
                {"A": 1.0},
                {"B": 1.0},
                ArrheniusRatePlan(rate),
            ),
        ),
    ).prepare()
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(2, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray(((0.0,), (1.0e6,))))
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=system.component_names
    ).prepare()
    method = phx.discretization.FiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.RusanovFluxPlan(),
        viscous=phx.discretization.ViscousFluxPlan() if viscous else None,
    )
    dynamics = PreparedFiniteVolumeDynamics(
        system,
        discretization,
        method,
        phx.discretization.FiniteVolumeBoundarySet.periodic(("x",)),
    )
    runtime = PreparedFiniteVolumeRuntime(
        dynamics,
        FluxPositivityPlan(2, fallback_flux=RusanovFluxPlan()),
    )
    one_cell = system.primitive_to_conserved(jnp.asarray((0.7, 0.3, 0.0, 800.0)))
    conserved = jnp.broadcast_to(one_cell, discretization.state_shape)
    return system, mechanism, runtime, conserved


def _balance(runtime, mechanism, *, integration, subcycles=8, iterations=20):
    transport = phx.solver.prepare_balance_law_transport(runtime)
    process = phx.solver.ThermochemistryProcessPlan(
        mechanism,
        subcycles=subcycles,
        integration=integration,
        nonlinear_iterations=iterations,
        nonlinear_tolerance=1.0e-8,
    ).prepare(transport)
    return phx.solver.PreparedBalanceLawRuntime(transport, (process,))


def _advance(balance, runtime, conserved, step):
    transport_state = runtime.initialize_state(conserved, 0.0, step)
    state = balance.initialize_state(transport_state)
    return balance.advance_prescribed(state, 0.0, step)


def _average(result, shape):
    return result.runtime_state.transport_state.cell_average().reshape(shape)


def _invariants(system, state):
    species = state[..., : system.species_count]
    amount = species / system.thermodynamics.schema.molar_masses
    schema = system.thermodynamics.schema
    return (
        jnp.sum(species, axis=-1),
        schema.element_amount(amount),
        schema.charge_amount(amount),
        state[..., system.energy_index],
    )


def test_balance_law_thermochemistry_advances_and_preserves_invariants():
    system, mechanism, runtime, conserved = _problem()
    balance = _balance(runtime, mechanism, integration="explicit-subcycled", subcycles=16)
    result = _advance(balance, runtime, conserved, 0.05)
    after = _average(result, conserved.shape)

    assert bool(result.accepted)
    assert float(after[0, 0]) < float(conserved[0, 0])
    for before, advanced in zip(
        _invariants(system, conserved), _invariants(system, after), strict=True
    ):
        np.testing.assert_allclose(advanced, before, rtol=2.0e-6, atol=2.0e-6)


def test_fixed_iterative_trapezoidal_chemistry_accepts_stiff_positive_update():
    system, mechanism, runtime, conserved = _problem(rate=20.0)
    balance = _balance(
        runtime,
        mechanism,
        integration="iterative-trapezoidal",
        subcycles=1,
        iterations=32,
    )
    result = _advance(balance, runtime, conserved, 0.08)
    after = _average(result, conserved.shape)

    assert bool(result.accepted)
    assert jnp.all(after[..., : system.species_count] >= 0.0)
    assert float(after[0, 0]) < float(conserved[0, 0])
    np.testing.assert_allclose(
        after[..., system.energy_index],
        conserved[..., system.energy_index],
        rtol=0.0,
        atol=0.0,
    )


def test_failed_explicit_chemistry_rolls_back_complete_balance_state():
    _, mechanism, runtime, conserved = _problem(rate=1.0e6)
    balance = _balance(runtime, mechanism, integration="explicit-subcycled", subcycles=1)
    result = _advance(balance, runtime, conserved, 0.01)
    after = _average(result, conserved.shape)

    assert not bool(result.accepted)
    np.testing.assert_array_equal(after, conserved)


def test_thermochemistry_process_composes_with_viscous_mixture_transport():
    system, mechanism, runtime, conserved = _problem(viscous=True)
    balance = _balance(runtime, mechanism, integration="explicit-subcycled", subcycles=16)
    result = _advance(balance, runtime, conserved, 0.02)
    after = _average(result, conserved.shape)

    assert isinstance(system, HomogeneousMixtureCompressibleNavierStokesSystem)
    assert bool(result.accepted)
    assert float(after[0, 0]) < float(conserved[0, 0])
    np.testing.assert_array_equal(
        after[..., system.energy_index], conserved[..., system.energy_index]
    )
