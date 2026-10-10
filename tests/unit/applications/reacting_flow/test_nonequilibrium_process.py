from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.applications.reacting_flow import (
    GradientLengthKnudsenPlan,
    LandauTellerRelaxationPlan,
    ThermochemicalNonequilibriumProcessPlan,
)


# The balance runtime refuses any bit change to a component a process does not
# own, and transport of this spatially uniform periodic state is flux-free, so
# every SSPRK3 stage returns its base.  Unowned components therefore move only
# by the rounding of the content/average conversions and stage combinations
# applied to one value: 4 content products (initialization, two source-view
# writes, transport acceptance) at one rounding each; 4 average quotients (two
# source views, the transport read, the read-out) at up to two roundings each,
# because XLA may lower division by a broadcast volume as a reciprocal product;
# and at most 5 roundings on any path of the stage combinations 3/4 u + 1/4 u
# and fl(1/3) u + fl(2/3) v.  With n = 17 roundings and unit roundoff 2**-53,
# |change| <= gamma_n |value|.
_UNIT_ROUNDOFF = 2.0**-53
_FLUX_FREE_RTOL = 17 * _UNIT_ROUNDOFF / (1.0 - 17 * _UNIT_ROUNDOFF)


def _problem(*, chemistry: Any = False) -> Any:
    schema = phx.equations.ChemicalSpeciesSchema.from_unique_species(
        ("A2", "A"),
        (phx.equations.ChemicalPhaseKind.GAS,) * 2,
        jnp.asarray((0.028, 0.014)),
        ("A",),
        jnp.asarray(((2, 1),), dtype=jnp.int32),
        jnp.zeros((2,), dtype=jnp.int32),
        gas_standard_pressure=1.0e5,
    )
    heavy = phx.equations.PolynomialSpeciesThermodynamicsPlan(
        schema,
        jnp.asarray(
            (
                2.5 * phx.equations.UNIVERSAL_GAS_CONSTANT,
                1.5 * phx.equations.UNIVERSAL_GAS_CONSTANT,
            )
        ),
        jnp.asarray((0.0, 2.0e4)),
        reference_temperature=300.0,
        minimum_temperature=100.0,
        maximum_temperature=10000.0,
    )
    modes = phx.equations.ThermalModeSchema(
        schema,
        (
            phx.equations.ThermalModeSpec(
                "vibrational-electronic",
                jnp.asarray((3390.0, 5000.0)),
                minimum_temperature=100.0,
                maximum_temperature=20000.0,
            ),
        ),
    )
    thermodynamics = phx.equations.TwoTemperatureThermodynamicsPlan(heavy, modes)
    system = phx.equations.TwoTemperatureMixtureEulerSystem(thermodynamics, 1)
    mechanism = None
    if chemistry:
        mechanism = phx.equations.ChemicalMechanismIR(
            "A2-dissociation-test",
            schema,
            heavy,
            (
                phx.equations.ChemicalReactionSpec(
                    "A2->2A",
                    {"A2": 1.0},
                    {"A": 2.0},
                    phx.equations.ArrheniusRatePlan(0.1),
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
    dynamics = phx.discretization.PreparedFiniteVolumeDynamics(
        system,
        discretization,
        phx.discretization.FiniteVolumeMethodPlan(
            phx.discretization.PiecewiseConstantReconstruction(),
            phx.discretization.HLLFluxPlan(),
        ),
        phx.discretization.FiniteVolumeBoundarySet.periodic(("x",)),
    )
    runtime = phx.solver.PreparedFiniteVolumeRuntime(
        dynamics,
        phx.discretization.FluxPositivityPlan(
            2, fallback_flux=phx.discretization.HLLFluxPlan()
        ),
    )
    one_cell = system.primitive_to_conserved(jnp.asarray((0.7, 0.1, 0.0, 2000.0, 500.0)))
    state = jnp.broadcast_to(one_cell, discretization.state_shape)
    return system, mechanism, runtime, state


def _advance(*, chemistry: Any = False) -> Any:
    system, mechanism, runtime, state = _problem(chemistry=chemistry)
    transport = phx.solver.prepare_balance_law_transport(runtime)
    process = ThermochemicalNonequilibriumProcessPlan(
        LandauTellerRelaxationPlan(jnp.asarray((0.01,))),
        mechanism,
        subcycles=4,
        nonlinear_iterations=8,
    ).prepare(transport)
    balance = phx.solver.PreparedBalanceLawRuntime(transport, (process,))
    runtime_state = runtime.initialize_state(state, 0.0, 1.0e-3)
    result = balance.advance_prescribed(
        balance.initialize_state(runtime_state), 0.0, 1.0e-3
    )
    # ty: ignore[unresolved-attribute]
    after = result.runtime_state.transport_state.cell_average().reshape(state.shape)
    return system, state, result, after


def test_nonequilibrium_process_scenario_1() -> None:
    system, before, result, after = _advance()
    assert bool(result.accepted)
    np.testing.assert_allclose(
        after[..., : system.species_count],
        before[..., : system.species_count],
        rtol=_FLUX_FREE_RTOL,
        atol=0.0,
    )
    assert jnp.all(after[..., system.mode_slice] > before[..., system.mode_slice])
    np.testing.assert_allclose(
        after[..., system.energy_index],
        before[..., system.energy_index],
        rtol=_FLUX_FREE_RTOL,
        atol=0.0,
    )
    assert jnp.all(system.mode_temperatures(after) > system.mode_temperatures(before))
    system, before, result, after = _advance(chemistry=True)
    assert bool(result.accepted)
    assert jnp.all(after[..., 0] < before[..., 0])
    before_species = before[..., : system.species_count]
    after_species = after[..., : system.species_count]
    np.testing.assert_allclose(
        jnp.sum(after_species, axis=-1),
        jnp.sum(before_species, axis=-1),
        rtol=2.0e-6,
        atol=2.0e-6,
    )
    np.testing.assert_allclose(
        after[..., system.energy_index],
        before[..., system.energy_index],
        rtol=_FLUX_FREE_RTOL,
        atol=0.0,
    )


def test_gradient_length_knudsen_reports_modes_and_hysteretic_recommendation() -> None:
    euler, _, _, _ = _problem()
    system = phx.equations.TwoTemperatureMixtureNavierStokesSystem(
        euler.thermodynamics,
        phx.equations.ConstantTransport(2.0e-5, 0.03),
        1,
        # ty: ignore[invalid-argument-type]
        mode_diffusivities=(4.0e-5,),
    )

    def state_at(coordinate: Any) -> Any:
        return system.primitive_to_conserved(
            jnp.asarray(
                (
                    0.7 + 0.05 * coordinate,
                    0.1 - 0.05 * coordinate,
                    100.0 + 3.0 * coordinate,
                    1800.0 + 20.0 * coordinate,
                    700.0 + 40.0 * coordinate,
                )
            )
        )

    state = state_at(jnp.asarray(0.0))
    gradient = jax.jacfwd(state_at)(jnp.asarray(0.0))[..., None]
    plan = GradientLengthKnudsenPlan(enter_threshold=1.0e-12, leave_threshold=5.0e-13)
    evidence, hysteresis = plan.evaluate(system, state, gradient)

    assert bool(evidence.header.eligible)
    assert evidence.mean_free_path > 0.0
    assert evidence.mode_temperature_knudsen.shape == (1,)
    assert bool(evidence.kinetic_recommended)
    assert bool(hysteresis.kinetic_recommended)
