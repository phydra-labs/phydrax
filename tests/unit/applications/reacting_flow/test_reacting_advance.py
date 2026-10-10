#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import equinox as eqx
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


# Thermochemistry never writes energy (the balance runtime refuses any bit change
# to a component a process does not own), and transport of this spatially
# uniform periodic state is flux-free, so every SSPRK3 stage returns its base.
# Energy therefore moves only by the rounding of the content/average
# conversions and stage combinations applied to one value: 4 content products
# (initialization, two source-view writes, transport acceptance) at one rounding
# each; 4 average quotients (two source views, the transport read, the
# read-out) at up to two roundings each, because XLA may lower division by a
# broadcast volume as a reciprocal product; and at most 5 roundings on any path
# of the stage combinations 3/4 u + 1/4 u and fl(1/3) u + fl(2/3) v.  With
# n = 17 roundings and unit roundoff 2**-53, |change| <= gamma_n |energy|.
_UNIT_ROUNDOFF = 2.0**-53
_FLUX_FREE_ENERGY_RTOL = 17 * _UNIT_ROUNDOFF / (1.0 - 17 * _UNIT_ROUNDOFF)


def _problem(rate: Any = 1.0, *, viscous: Any = False) -> Any:
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


def _balance(
    runtime: Any,
    mechanism: Any,
    *,
    integration: Any,
    subcycles: Any = 8,
    iterations: Any = 20,
) -> Any:
    transport = phx.solver.prepare_balance_law_transport(runtime)
    process = phx.solver.ThermochemistryProcessPlan(
        mechanism,
        subcycles=subcycles,
        integration=integration,
        nonlinear_iterations=iterations,
        nonlinear_tolerance=1.0e-8,
    ).prepare(transport)
    return phx.solver.PreparedBalanceLawRuntime(transport, (process,))


def _advance(balance: Any, runtime: Any, conserved: Any, step: Any) -> Any:
    transport_state = runtime.initialize_state(conserved, 0.0, step)
    state = balance.initialize_state(transport_state)
    return balance.advance_prescribed(state, 0.0, step)


def _average(result: Any, shape: Any) -> Any:
    return result.runtime_state.transport_state.cell_average().reshape(shape)


def _invariants(system: Any, state: Any) -> Any:
    species = state[..., : system.species_count]
    amount = species / system.thermodynamics.schema.molar_masses
    schema = system.thermodynamics.schema
    return (
        jnp.sum(species, axis=-1),
        schema.element_amount(amount),
        schema.charge_amount(amount),
        state[..., system.energy_index],
    )


def test_reacting_advance_scenario_1() -> None:
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
        rtol=_FLUX_FREE_ENERGY_RTOL,
        atol=0.0,
    )
    _, mechanism, runtime, conserved = _problem(rate=1.0e6)
    balance = _balance(runtime, mechanism, integration="explicit-subcycled", subcycles=1)
    result = _advance(balance, runtime, conserved, 0.01)
    after = _average(result, conserved.shape)
    incoming = runtime.initialize_state(conserved, 0.0, 0.01).cell_average()

    assert not bool(result.accepted)
    np.testing.assert_array_equal(after, incoming.reshape(conserved.shape))


def test_thermochemistry_process_composes_with_viscous_mixture_transport() -> None:
    system, mechanism, runtime, conserved = _problem(viscous=True)
    balance = _balance(runtime, mechanism, integration="explicit-subcycled", subcycles=16)
    result = _advance(balance, runtime, conserved, 0.02)
    after = _average(result, conserved.shape)

    assert isinstance(system, HomogeneousMixtureCompressibleNavierStokesSystem)
    assert bool(result.accepted)
    assert float(after[0, 0]) < float(conserved[0, 0])
    np.testing.assert_allclose(
        after[..., system.energy_index],
        conserved[..., system.energy_index],
        rtol=_FLUX_FREE_ENERGY_RTOL,
        atol=0.0,
    )


def test_compiled_reacting_rollout_matches_eager_step() -> None:
    _, mechanism, runtime, conserved = _problem()
    balance = _balance(runtime, mechanism, integration="explicit-subcycled", subcycles=16)
    initial = balance.initialize_state(runtime.initialize_state(conserved, 0.0, 0.05))
    plan = phx.solver.ScheduledBalanceLawRolloutPlan(
        balance, phx.discretization.TemporalMesh(jnp.asarray((0.0, 0.05)))
    )
    eager = balance.advance_prescribed(initial, 0.0, 0.05)
    compiled = eqx.filter_jit(lambda rollout, state: rollout.rollout(state))(
        plan, initial
    )

    assert bool(eager.accepted)
    assert bool(compiled.accepted[0])
    assert int(compiled.statuses[0]) == int(eager.status)
    np.testing.assert_array_equal(
        compiled.final_state.transport_state.content_state.conservative_content,
        eager.runtime_state.transport_state.content_state.conservative_content,
    )
