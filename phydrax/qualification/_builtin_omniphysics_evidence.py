#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Executable independent controls and refinement campaigns for omniphysics APIs."""

from __future__ import annotations

import platform

import jax
import jax.numpy as jnp
import numpy as np

from ._omniphysics_evidence import (
    ApplicationValidationEvidence,
    HardwareProviderEvidence,
    NumericalControlEvidence,
    OmniphysicsQualificationEvidence,
    RefinementCampaignEvidence,
)


def _float(value) -> float:
    return float(np.asarray(value))


def _control(
    family, name, observed, reference=0.0, tolerance=1e-10, source="closed-form"
):
    return NumericalControlEvidence(
        family,
        name,
        _float(observed),
        float(reference),
        float(tolerance),
        0.0,
        source,
    )


def _thermal_refinement() -> RefinementCampaignEvidence:
    from phydrax.applications.additive_manufacturing import implicit_thermal_step

    resolutions = (0.25, 0.125, 0.0625)
    errors = []
    initial = jnp.asarray((1.0, -1.0))
    capacity = jnp.ones(2)
    laplacian = jnp.asarray(((1.0, -1.0), (-1.0, 1.0)))
    exact = jnp.exp(-2.0) * initial
    for step_size in resolutions:
        value = initial
        for _ in range(round(1 / step_size)):
            value = implicit_thermal_step(
                value, capacity, laplacian, jnp.zeros(2), step_size
            ).value
        errors.append(_float(jnp.sqrt(jnp.sum((value - exact) ** 2))))
    return RefinementCampaignEvidence(
        "implicit-thermal-decay",
        ("materials", "manufacturing", "thermal"),
        resolutions,
        tuple(errors),
        0.7,
        "two-cell diffusion eigenmode exp(-2t)",
    )


def _structural_refinement() -> RefinementCampaignEvidence:
    from phydrax.structural_dynamics import LinearStructuralSystem, StructuralDynamicState

    resolutions = (0.2, 0.1, 0.05)
    errors = []
    system = LinearStructuralSystem.create(
        jnp.asarray(((1.0,),)), jnp.zeros((1, 1)), jnp.asarray(((1.0,),))
    )
    exact_displacement = np.cos(1.0)
    exact_velocity = -np.sin(1.0)
    for step_size in resolutions:
        state = StructuralDynamicState(
            jnp.asarray((1.0,)), jnp.asarray((0.0,)), jnp.asarray((-1.0,))
        )
        for _ in range(round(1 / step_size)):
            state = system.newmark_step(state, jnp.zeros(1), step_size).state
        error = jnp.sqrt(
            (state.displacement[0] - exact_displacement) ** 2
            + (state.velocity[0] - exact_velocity) ** 2
        )
        errors.append(_float(error))
    return RefinementCampaignEvidence(
        "newmark-undamped-oscillator",
        (
            "structural-dynamics",
            "correlation",
            "acoustics",
            "smart-materials",
            "optomechanics",
        ),
        resolutions,
        tuple(errors),
        1.8,
        "unit oscillator q(t)=cos(t)",
    )


def _constitutive_refinement() -> RefinementCampaignEvidence:
    from phydrax.rheology import SpatialConformationSolver, ViscoelasticLaw

    resolutions = (0.2, 0.1, 0.05)
    errors = []
    law = ViscoelasticLaw("oldroyd-b", 1.0, 1.0)
    solver = SpatialConformationSolver.create(jnp.ones(1), jnp.zeros((1, 1)), law)
    exact = 1 + np.exp(-1.0)
    for step_size in resolutions:
        conformation = jnp.asarray([[[2.0]]])
        for _ in range(round(1 / step_size)):
            conformation = solver.advance(
                conformation, jnp.zeros((1, 1, 1)), step_size
            ).conformation
        errors.append(abs(_float(conformation[0, 0, 0]) - exact))
    return RefinementCampaignEvidence(
        "oldroyd-relaxation",
        ("rheology", "electrohydrodynamics", "interfacial-transport"),
        resolutions,
        tuple(errors),
        0.8,
        "Oldroyd-B homogeneous A(t)=I+(A0-I)exp(-t/lambda)",
    )


def _controls_and_applications():
    from phydrax import (
        acoustics,
        chemo_mechanics,
        correlation,
        electrochemistry,
        electrohydrodynamics,
        frequency,
        interfacial_transport,
        manufacturing,
        materials,
        membranes,
        optomechanics,
        phoresis,
        plasma,
        population_balance,
        process_systems,
        rheology,
        smart_materials,
        structural_dynamics,
        surface_chemistry,
        system_modeling,
        thermal_systems,
        tribology,
    )
    from phydrax.applications import flight_dynamics, reservoir

    controls = []

    material_state = materials.MaterialState(
        jnp.asarray((300.0, 400.0)),
        jnp.asarray((1e5, 1e5)),
        jnp.asarray(((1.0, 0.0), (0.0, 1.0))),
    )
    material_field = materials.SpatialMaterialField.create(
        jnp.asarray(((0.0,), (1.0,))), jnp.ones(2), material_state
    )
    material_transfer = materials.ConservativeMaterialTransfer.create(
        jnp.asarray(((0.75, 0.25), (0.25, 0.75))), jnp.ones(2), jnp.ones(2)
    )
    transferred = material_field.transfer(
        jnp.asarray(((0.25,), (0.75,))), material_transfer
    )
    controls.append(
        _control(
            "materials",
            "weighted-phase-transfer",
            jnp.max(
                jnp.abs(
                    material_field.integral(material_state.phase_fractions)
                    - transferred.integral(transferred.state.phase_fractions)
                )
            ),
            source="finite-volume integral identity",
        )
    )

    event = manufacturing.ToolpathEvent(
        "control", "deposit", 0.0, 1.0, "machine", (0.0,), (1.0,), 10.0, 2.0
    )
    runtime = manufacturing.ManufacturingRuntime.create(
        manufacturing.ProcessSchedule.create((event,)),
        jnp.asarray(((0.0,), (1.0,))),
        jnp.ones(2),
        1.0,
    )
    runtime_step = runtime.advance(
        manufacturing.ManufacturingRuntimeState.initialize(2), 1.0
    )
    controls.append(
        _control(
            "manufacturing",
            "process-mass-energy-ledger",
            jnp.maximum(
                jnp.abs(runtime_step.mass_balance_residual_kg),
                jnp.abs(runtime_step.energy_balance_residual_j),
            ),
            source="integrated scheduled source",
        )
    )

    frequency_result = frequency.CompiledFrequencySystem.create(
        jnp.asarray(((1.0,),)), jnp.asarray(((0.2,),)), jnp.asarray(((4.0,),))
    ).solve(jnp.asarray((1.0,)), jnp.asarray((1.0,)))
    controls.append(
        _control(
            "frequency",
            "single-dof-complex-response",
            jnp.abs(frequency_result.response[0, 0] - 1 / (3 + 0.2j)),
            source="analytic dynamic stiffness inverse",
        )
    )

    population_solver = population_balance.ConservativeSectionalSolver.create(
        jnp.asarray((1.0, 2.0, 4.0)), jnp.ones((3, 3))
    )
    population_state = population_balance.SectionalPopulationState.create(
        jnp.asarray((1.0, 0.0, 0.25))
    )
    population_step = population_solver.advance(population_state, 0.1)
    controls.append(
        _control(
            "population-balance",
            "aggregation-first-moment",
            population_step.first_moment_residual,
            source="Smoluchowski first-moment invariant with overflow reservoir",
        )
    )

    connector_type = system_modeling.ConnectorType.create(
        "electric",
        (
            system_modeling.ConnectorVariable("potential", "across", "V"),
            system_modeling.ConnectorVariable("current", "through", "A"),
        ),
    )
    acausal = system_modeling.AcausalSystem.create(
        (
            system_modeling.Connector("a", connector_type),
            system_modeling.Connector("b", connector_type),
        ),
        (system_modeling.ConnectionSet.create(("a", "b")),),
    )
    compiled = system_modeling.compile_linear_acausal_system(
        acausal,
        jnp.asarray(((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, -2.0))),
        jnp.asarray((10.0, 0.0)),
    ).solve()
    controls.append(
        _control("system-modeling", "acausal-connection-residual", compiled.residual_norm)
    )

    law = rheology.ViscoelasticLaw("oldroyd-b", 1.0, 1.0)
    rheology_step = rheology.SpatialConformationSolver.create(
        jnp.ones(1), jnp.zeros((1, 1)), law
    ).advance(jnp.ones((1, 1, 1)), jnp.zeros((1, 1, 1)), 0.1)
    controls.append(
        _control(
            "rheology",
            "equilibrium-conformation-invariance",
            jnp.max(jnp.abs(rheology_step.conformation - 1)),
        )
    )

    surface = interfacial_transport.CoupledBulkSurfaceTransport.create(
        jnp.ones(1),
        jnp.ones(1),
        jnp.zeros((1, 1)),
        jnp.ones((1, 1)),
        interfacial_transport.AdsorptionKinetics(0.1, 0.0, 1.0),
    ).advance(jnp.ones(1), jnp.zeros(1), 0.1)
    controls.append(
        _control(
            "interfacial-transport",
            "bulk-surface-species-balance",
            surface.total_mole_balance_residual,
        )
    )

    structural = structural_dynamics.LinearStructuralSystem.create(
        jnp.eye(2), jnp.zeros((2, 2)), jnp.diag(jnp.asarray((4.0, 9.0)))
    ).modal_analysis()
    controls.append(
        _control(
            "structural-dynamics",
            "diagonal-generalized-modes",
            jnp.max(
                jnp.abs(structural.angular_frequencies_rad_s - jnp.asarray((2.0, 3.0)))
            ),
        )
    )
    modal = correlation.correlate_modes(
        jnp.asarray((10.0, 20.0)),
        jnp.eye(2),
        jnp.asarray((20.0, 10.0)),
        jnp.asarray(((0.0, 1.0), (1.0, 0.0))),
        jnp.eye(2),
    )
    controls.append(
        _control(
            "correlation",
            "permuted-mode-mac",
            jnp.min(modal.matched_modal_assurance),
            1.0,
        )
    )

    ehd_solver = electrohydrodynamics.CoupledElectrohydrodynamicSolver.create(
        jnp.ones(2),
        jnp.eye(2),
        jnp.eye(2),
        jnp.asarray(((-1.0, 1.0), (1.0, -1.0))),
        jnp.eye(2),
        jnp.ones(2),
        jnp.ones(2),
        1,
    )
    ehd_step = ehd_solver.advance(
        electrohydrodynamics.ElectrohydrodynamicState(
            jnp.asarray((1.0, -1.0)), jnp.zeros(2), jnp.zeros((2, 1))
        ),
        jnp.zeros(2),
        0.1,
    )
    controls.append(
        _control(
            "electrohydrodynamics",
            "closed-domain-charge",
            ehd_step.charge_balance_residual_c,
        )
    )

    mobility, _ = phoresis.HydrodynamicPhoreticSolver.create(
        1.0, jnp.asarray((0.5, 0.5)), 3
    ).mobility(jnp.asarray(((0.0, 0.0, 0.0), (3.0, 0.0, 0.0))))
    controls.append(
        _control(
            "phoresis", "mobility-reciprocity", jnp.max(jnp.abs(mobility - mobility.T))
        )
    )

    piezo = smart_materials.SpatialPiezoelectricSystem.create(
        jnp.asarray(((2.0,),)), jnp.asarray(((3.0,),)), jnp.asarray(((0.5,),))
    ).solve(jnp.asarray((1.0,)), jnp.asarray((0.2,)))
    controls.append(
        _control("smart-materials", "reciprocal-block-residual", piezo.residual_norm)
    )

    chemo = chemo_mechanics.SpatialChemoMechanicalSystem.create(
        jnp.asarray(((2.0,),)),
        jnp.eye(2),
        jnp.asarray(((0.1, 0.1),)),
        jnp.asarray(((1.0, -1.0), (-1.0, 1.0))),
        jnp.ones(2),
    ).advance(jnp.asarray((0.2, 0.8)), jnp.zeros(1), jnp.zeros(2), 0.1)
    controls.append(
        _control("chemo-mechanics", "closed-species-balance", chemo.mass_balance_residual)
    )

    ehl = tribology.MassConservingEHLSolver.create(
        jnp.ones(3), jnp.ones(3), jnp.zeros((3, 3)), 1.0, 0.0, bulk_modulus_pa=1000.0
    ).advance(tribology.EHLState(jnp.asarray((1.1, 0.5, 1.0)), jnp.zeros(3)), 1e-4)
    controls.append(
        _control("tribology", "elrod-content-balance", ehl.mass_balance_residual_m2)
    )

    radiation = thermal_systems.DiffuseGrayEnclosure.create(
        jnp.ones(2), jnp.ones(2), jnp.asarray(((0.0, 1.0), (1.0, 0.0)))
    ).solve(jnp.asarray((300.0, 400.0)))
    controls.append(
        _control("thermal", "enclosure-energy-balance", radiation.enclosure_balance_w)
    )

    membrane = membranes.CrossflowMembraneModule.create(
        jnp.ones(2),
        jnp.asarray((1e-6, 0.5e-6)),
        jnp.full(2, 2e5),
        jnp.full(2, 1e5),
    ).solve(jnp.asarray((10.0, 10.0)), jnp.asarray((1.0, 1.0)))
    controls.append(
        _control(
            "membranes",
            "module-species-balance",
            jnp.max(jnp.abs(membrane.species_balance_residual_mol_s)),
        )
    )

    catalyst = surface_chemistry.SegmentedCatalyticReactor.create(
        jnp.ones(2),
        jnp.asarray(((-1.0, 1.0),)),
        jnp.asarray(((1.0, 0.0),)),
        jnp.asarray((0.1,)),
        jnp.asarray((-1000.0,)),
        jnp.asarray(((1.0, 1.0),)),
        1.0,
        100.0,
    ).solve(jnp.asarray((1.0, 0.0)), 300.0)
    controls.append(
        _control(
            "surface-chemistry",
            "reactor-element-balance",
            jnp.max(jnp.abs(catalyst.conserved_quantity_residual)),
        )
    )

    stop = optomechanics.SpatialOptomechanicalSystem.create(
        jnp.eye(2),
        jnp.eye(2),
        jnp.zeros((2, 2)),
        jnp.eye(2),
        jnp.zeros((2, 2)),
        jnp.ones(2),
    ).solve(jnp.asarray((1.0, 2.0)), jnp.asarray((0.5, 0.0)))
    controls.append(
        _control(
            "optomechanics",
            "stop-field-residual",
            jnp.maximum(stop.thermal_residual_norm, stop.mechanical_residual_norm),
        )
    )

    acoustic = acoustics.VibroacousticSystem.create(
        jnp.ones((1, 1)),
        jnp.ones((1, 1)) * 0.1,
        jnp.ones((1, 1)) * 4,
        jnp.ones((1, 1)),
        jnp.ones((1, 1)) * 0.2,
        jnp.ones((1, 1)) * 9,
        jnp.ones((1, 1)) * 0.5,
    ).solve(1.0, jnp.ones(1), jnp.zeros(1))
    controls.append(
        _control(
            "acoustics",
            "vibroacoustic-block-residual",
            jnp.maximum(
                acoustic.structural_residual_norm, acoustic.acoustic_residual_norm
            ),
        )
    )

    electrode = electrochemistry.PorousElectrodeSystem.create(
        jnp.ones(2),
        jnp.zeros((1, 2, 2)),
        jnp.eye(2),
        jnp.asarray((-1.0,)),
        jnp.ones(2),
        jnp.ones(2),
        jnp.zeros(2),
        jnp.full(2, 300.0),
        1,
    ).advance(
        electrochemistry.PorousElectrodeState(jnp.ones((2, 1)), jnp.zeros(2)),
        jnp.asarray((0.1, 0.1)),
        0.1,
    )
    controls.append(
        _control(
            "electrochemistry",
            "porous-current-species-residual",
            jnp.maximum(
                electrode.current_residual_norm_a,
                jnp.max(jnp.abs(electrode.species_balance_residual_mol)),
            ),
        )
    )

    flowsheet = process_systems.EquationOrientedFlowsheet.create(
        lambda value: jnp.asarray((value[0] + value[1] - 10, value[1] - 0.25 * value[0])),
        jnp.asarray((10.0, 2.0)),
        jnp.asarray((10.0, 2.0)),
        lower_bounds=jnp.zeros(2),
    ).solve(jnp.asarray((5.0, 1.0)))
    controls.append(
        _control(
            "process-systems", "recycle-equation-residual", flowsheet.scaled_residual_norm
        )
    )

    ded_application = ApplicationValidationEvidence(
        "additive-manufacturing",
        "two-cell-conservative-source",
        (
            (
                "scheduled-energy-j",
                _float(jnp.sum(runtime_step.state.supplied_energy_j)),
                10.0,
                1e-10,
            ),
            (
                "deposited-mass-kg",
                _float(jnp.sum(runtime_step.state.deposited_mass_kg)),
                2.0,
                1e-10,
            ),
        ),
        "closed-form constant-rate integration",
    )
    maxwell_pic = plasma.ElectrostaticPIC1D(1.0, 16, 1.0).advance(
        plasma.ElectrostaticPICState(
            jnp.asarray((0.25, 0.75)), jnp.zeros(2), jnp.asarray(0.0)
        ),
        jnp.asarray((1.0, -1.0)),
        jnp.ones(2),
        jnp.ones(2),
        1e-3,
    )
    plasma_application = ApplicationValidationEvidence(
        "electrostatic-pic",
        "periodic-neutral-dipole",
        (
            (
                "neutralized-charge-c",
                _float(maxwell_pic.neutralized_charge_residual_c),
                0.0,
                1e-12,
            ),
            (
                "gauss-residual-c-m",
                _float(maxwell_pic.gauss_residual_norm_c_m),
                0.0,
                1e-10,
            ),
        ),
        "spectral periodic Poisson identity",
    )
    reservoir_step = reservoir.ReservoirPressureSystem.create(
        jnp.asarray((1e-6, 1e-6)),
        jnp.asarray(((1e-8, -1e-8), (-1e-8, 1e-8))),
    ).advance(
        reservoir.ReservoirPressureState(
            jnp.full(2, 1e7), jnp.asarray(0.0), jnp.asarray(0.0), jnp.asarray(0.0)
        ),
        jnp.zeros(2),
        jnp.asarray((1e-8, 0.0)),
        jnp.full(2, 9e6),
        100.0,
    )
    reservoir_application = ApplicationValidationEvidence(
        "reservoir-pressure",
        "closed-two-cell-well-balance",
        (
            (
                "volume-balance-m3",
                _float(reservoir_step.volume_balance_residual_m3),
                0.0,
                1e-12,
            ),
        ),
        "finite-volume material balance",
    )
    flight = flight_dynamics.RigidBodyFlightSystem.create(
        2.0, jnp.diag(jnp.asarray((1.0, 2.0, 3.0)))
    ).advance(
        flight_dynamics.FlightDynamicsState(
            jnp.zeros(3),
            jnp.zeros(3),
            jnp.asarray((1.0, 0.0, 0.0, 0.0)),
            jnp.zeros(3),
            jnp.asarray(0.0),
        ),
        jnp.asarray((0.0, 0.0, 2 * 9.80665)),
        jnp.zeros(3),
        0.1,
    )
    flight_application = ApplicationValidationEvidence(
        "flight-dynamics",
        "gravity-trim",
        (
            (
                "speed-m-s",
                _float(jnp.sqrt(jnp.sum(flight.state.velocity_body_m_s**2))),
                0.0,
                1e-12,
            ),
            ("quaternion-norm-error", _float(flight.quaternion_norm_error), 0.0, 1e-12),
        ),
        "Newtonian static trim",
    )
    return tuple(controls), (
        ded_application,
        plasma_application,
        reservoir_application,
        flight_application,
    )


def builtin_omniphysics_qualification_evidence() -> OmniphysicsQualificationEvidence:
    controls, applications = _controls_and_applications()
    devices = jax.devices()
    provider = HardwareProviderEvidence(
        f"jax-{jax.default_backend()}-{platform.machine()}",
        jax.default_backend(),
        jax.process_count(),
        1,
        len(devices),
        tuple(sorted({device.device_kind for device in devices})),
        "float64",
        True,
    )
    return OmniphysicsQualificationEvidence(
        controls,
        (_thermal_refinement(), _structural_refinement(), _constitutive_refinement()),
        (provider,),
        applications,
    )


__all__ = ["builtin_omniphysics_qualification_evidence"]
