import numpy as np

import phydrax as phx


def test_linear_dipole_surface_produces_finite_ir_line_strengths():
    units = phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
    positions = np.asarray(
        [[0.0, 0.0, 0.0], [0.95, 0.0, 0.0], [-0.24, 0.92, 0.0]],
        dtype=float,
    )
    structure = phx.atomistic.AtomicStructure(
        [8, 1, 1],
        positions,
        [15.999, 1.008, 1.008],
        units.scale,
        particle_ids=[11, 12, 13],
    )
    system = phx.atomistic.AtomisticSystemPlan.from_structure(
        structure, units, molecule_ids=[0, 0, 0]
    )
    state = phx.chemistry.MolecularElectronicStatePlan(0, 1)
    model = phx.chemistry.ElectronicModelChemistryPlan(
        phx.chemistry.ElectronicMethodPlan(
            phx.chemistry.ElectronicMethodFamily.CUSTOM_EXTERNAL,
            "analytic-dipole",
            phx.chemistry.ElectronicReferenceKind.RESTRICTED,
        )
    )
    request = phx.chemistry.ElectronicPropertyRequest(
        (
            phx.chemistry.ElectronicProperty.ENERGY,
            phx.chemistry.ElectronicProperty.FORCES,
            phx.chemistry.ElectronicProperty.DIPOLE,
        )
    )
    calculation = phx.chemistry.ElectronicCalculationPlan(system, state, model, request)
    charges = np.asarray([-0.8, 0.4, 0.4])

    def electronic(plan, coordinate, cell):
        del cell
        value = np.asarray(coordinate)
        delta = value - positions
        return phx.chemistry.make_electronic_evaluation(
            plan,
            "analytic-dipole-provider",
            value,
            np.sum(delta**2),
            forces=-2.0 * delta,
            dipole=np.sum(charges[:, None] * value, axis=0),
        )

    capabilities = phx.chemistry.ElectronicProviderCapabilities(
        (
            phx.chemistry.ElectronicProperty.ENERGY,
            phx.chemistry.ElectronicProperty.FORCES,
            phx.chemistry.ElectronicProperty.DIPOLE,
        ),
        (phx.chemistry.ElectronicReferenceKind.RESTRICTED,),
    )
    prepared = phx.chemistry.CallableElectronicProvider(
        electronic, "analytic-dipole-provider", capabilities
    ).prepare(calculation)

    def harmonic(coordinate, _):
        delta = np.asarray(coordinate) - positions
        return phx.chemistry.PotentialEnergySurfaceEvaluation(
            0.5 * np.sum(delta**2),
            -delta,
            None,
            True,
            provider_id="harmonic",
            source_result_id=phx.chemistry.electronic_geometry_id(system, coordinate),
        )

    surface = phx.chemistry.CallablePotentialEnergySurface(
        harmonic,
        system.system_id,
        units,
        "harmonic",
        phx.chemistry.PotentialEnergySurfaceCapabilities(),
    )
    hessian = phx.chemistry.MolecularHessianPlan(
        system, surface, displacement=1.0e-4, antisymmetry_tolerance=1.0e-10
    ).evaluate(structure)
    vibration = phx.chemistry.VibrationalAnalysisPlan(system).evaluate(structure, hessian)
    spectrum = phx.chemistry.IRSpectrumPlan(
        system,
        prepared,
        displacement=1.0e-4,
        maximum_wavenumber=5000.0,
        grid_size=501,
    ).evaluate(structure, vibration)

    assert bool(spectrum.successful)
    assert spectrum.wavenumber_unit == phx.units.INVERSE_CENTIMETER
    assert np.all(np.asarray(spectrum.line_strengths) >= 0.0)
    assert float(np.max(np.asarray(spectrum.intensity))) > 0.0
