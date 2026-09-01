#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def test_open_boundary_absorbs_particle_and_closes_surface_ledgers():
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(2), jnp.ones((2,)), ambient_dimension=1
    ).prepare()
    population_plan = phx.discretization.ParticlePopulationPlan(particles)
    population = population_plan.initialize()
    state = phx.discretization.pic.PICParticleState(
        jnp.asarray([[0.2], [0.8]]),
        jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    plan = phx.discretization.pic.PICOpenBoundaryPlan(
        jnp.asarray([0.0]),
        jnp.asarray([1.0]),
        kinds=(
            phx.discretization.pic.PICBoundaryKind.ABSORB,
            phx.discretization.pic.PICBoundaryKind.ABSORB,
        ),
    )
    result = plan.apply(
        population_plan,
        population,
        state,
        jnp.asarray([[0.2], [1.2]]),
        jnp.asarray([-1.0, -1.0]),
        plan.initialize_surface(),
    )
    assert result.successful
    assert result.hit_mask[1]
    assert not result.accepted_population.active[1]
    assert jnp.sum(result.boundary_charge_flux) == -1.0
    assert jnp.sum(result.boundary_mass_flux) == 1.0


def test_field_ionization_is_charge_neutral_when_event_occurs():
    ion_support = phx.discretization.ParticleSetPlan(
        jnp.arange(2), jnp.ones((2,)), ambient_dimension=3
    ).prepare()
    electron_support = phx.discretization.ParticleSetPlan(
        jnp.arange(10, 12), jnp.ones((2,)), ambient_dimension=3
    ).prepare()
    ion_population = phx.discretization.ParticlePopulationPlan(
        ion_support
    ).initialize(
        active_mask=jnp.asarray([True, False]), masses=jnp.asarray([1.0, 0.0])
    )
    electron_population_plan = phx.discretization.ParticlePopulationPlan(
        electron_support
    )
    electron_population = electron_population_plan.initialize(
        active_mask=jnp.asarray([False, False]), masses=jnp.zeros((2,))
    )
    ion_model = phx.discretization.pic.PICChargeModelPlan(
        1.0,
        "ions",
        minimum_charge_number=0,
        maximum_charge_number=2,
        initial_charge_number=0,
    )
    electron_model = phx.discretization.pic.PICChargeModelPlan(
        -1.0,
        "electrons",
        minimum_charge_number=1,
        maximum_charge_number=1,
        initial_charge_number=1,
    )
    ions = phx.discretization.pic.PICParticleState(
        jnp.asarray([[0.2, 0.2, 0.2], [0.0, 0.0, 0.0]]), jnp.zeros((2, 3))
    )
    electrons = phx.discretization.pic.PICParticleState(
        jnp.zeros((2, 3)), jnp.zeros((2, 3))
    )
    result = phx.discretization.pic.ionization.FieldIonizationPlan(
        1.0,
        field_power=1.0,
        ionization_energy=0.1,
        maximum_probability=0.25,
        maximum_events=1,
    ).apply(
        ion_model,
        ion_population,
        ion_model.initialize(ion_population),
        ions,
        jnp.asarray([[1.0e6, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        electron_model,
        electron_population_plan,
        electron_population,
        electron_model.initialize(electron_population),
        electrons,
        jnp.asarray([0, 1], dtype=jnp.uint32),
        1.0e-7,
        1,
    )
    assert result.successful
    assert jnp.abs(result.charge_defect) < 1e-12


def test_zero_response_semi_implicit_pic_is_stationary_and_constrained():
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformCellAxisSpec(2, periodic=True) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
    bridge = phx.discretization.StructuredCochainBridge(grid)
    support = phx.discretization.ParticleSetPlan(
        jnp.arange(1), jnp.ones((1,)), ambient_dimension=3
    ).prepare()
    charged = phx.discretization.ChargedParticlePlan(
        -jnp.ones((1,)), "route"
    ).prepare(support)
    transfer = phx.discretization.pic.PICParticleCochainTransferPlan(
        bridge
    ).prepare(charged)
    population = phx.discretization.ParticlePopulationPlan(support).initialize()
    charge_model = phx.discretization.pic.PICChargeModelPlan(
        -1.0,
        "neutralized",
        minimum_charge_number=0,
        maximum_charge_number=1,
        initial_charge_number=0,
    )
    maxwell = phx.solver.CompatibleMaxwellPlan(
        bridge,
        sources=(phx.solver.PICMaxwellCurrentSourcePlan(),),
        plan_id="semi-implicit-test",
    ).prepare()
    plan = phx.solver.SemiImplicitPICPlan(
        maxwell, transfer, charge_model, tolerance=1e-7
    )
    state = phx.solver.SemiImplicitPICState(
        phx.discretization.pic.PICParticleState(
            jnp.asarray([[0.25, 0.25, 0.25]]), jnp.zeros((1, 3))
        ),
        population,
        charge_model.initialize(population),
        maxwell.initialize(),
        jnp.asarray(0.0),
    )
    result = plan.step(state, 1e-3)
    assert result.successful
    assert result.diagnostics.gauss_defect < 1e-8
    assert result.diagnostics.magnetic_defect < 1e-8
