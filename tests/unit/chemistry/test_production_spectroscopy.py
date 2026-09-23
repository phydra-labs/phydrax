import numpy as np

import phydrax as phx


def test_tda_tracking_preserves_parent_root_identity_across_reordering():
    dipole_unit = phx.units.derived_unit(
        "e*bohr", ((phx.units.ELEMENTARY_CHARGE, 1), (phx.units.BOHR, 1))
    )
    manifold_plan = phx.chemistry.ExcitedStateManifoldPlan(
        2,
        degeneracy_absolute=1.0e-5,
    )
    parent = phx.chemistry.TammDancoffPlan(
        manifold_plan,
        [[0.2, 0.0], [0.0, 0.3]],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        -1.0,
        phx.units.HARTREE,
        dipole_unit,
    ).solve()
    candidate_plan = phx.chemistry.TammDancoffPlan(
        manifold_plan,
        [[0.31, 0.0], [0.0, 0.19]],
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        -1.0,
        phx.units.HARTREE,
        dipole_unit,
    )
    raw_candidate = candidate_plan.solve()
    tracking = phx.chemistry.track_excited_states(parent, raw_candidate)
    candidate = candidate_plan.solve(parent=parent)

    assert bool(candidate.successful)
    np.testing.assert_allclose(candidate.excitation_energies, [0.31, 0.19])
    assert bool(tracking.successful)
    np.testing.assert_array_equal(tracking.permutation, [1, 0])


def test_tracked_manifolds_produce_antisymmetric_nonadiabatic_coupling():
    dipole_unit = phx.units.derived_unit(
        "e*bohr", ((phx.units.ELEMENTARY_CHARGE, 1), (phx.units.BOHR, 1))
    )
    plan = phx.chemistry.ExcitedStateManifoldPlan(2)
    transition = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    reference = phx.chemistry.TammDancoffPlan(
        plan,
        [[0.2, 0.0], [0.0, 0.4]],
        transition,
        -1.0,
        phx.units.HARTREE,
        dipole_unit,
    ).solve()
    angle = 1.0e-3
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    diagonal = np.diag([0.2, 0.4])
    plus = phx.chemistry.TammDancoffPlan(
        plan,
        rotation @ diagonal @ rotation.T,
        transition,
        -1.0,
        phx.units.HARTREE,
        dipole_unit,
    ).solve(parent=reference)
    minus = phx.chemistry.TammDancoffPlan(
        plan,
        rotation.T @ diagonal @ rotation,
        transition,
        -1.0,
        phx.units.HARTREE,
        dipole_unit,
    ).solve(parent=reference)
    coupling = phx.chemistry.finite_difference_nonadiabatic_coupling(
        reference,
        plus,
        minus,
        0.01,
        phx.units.ANGSTROM,
    )

    assert bool(coupling.successful)
    np.testing.assert_allclose(
        coupling.derivative_couplings,
        -np.asarray(coupling.derivative_couplings).T,
        atol=1.0e-12,
    )
    assert coupling.energy_weighted_unit.dimension == (
        phx.units.ENERGY / phx.units.LENGTH
    )


def test_uv_visible_broadening_preserves_strength_on_each_supported_axis():
    dipole_unit = phx.units.derived_unit(
        "e*bohr", ((phx.units.ELEMENTARY_CHARGE, 1), (phx.units.BOHR, 1))
    )
    manifold = phx.chemistry.TammDancoffPlan(
        phx.chemistry.ExcitedStateManifoldPlan(1),
        [[0.2]],
        [[1.0, 0.0, 0.0]],
        -1.0,
        phx.units.HARTREE,
        dipole_unit,
    ).solve()
    configurations = (
        (phx.chemistry.SpectralAxis.ENERGY_EV, 1.0, 10.0, 0.1, 9001),
        (
            phx.chemistry.SpectralAxis.WAVENUMBER_CM1,
            10000.0,
            70000.0,
            500.0,
            6001,
        ),
        (
            phx.chemistry.SpectralAxis.WAVELENGTH_NM,
            100.0,
            500.0,
            2.0,
            4001,
        ),
    )
    for axis, lower, upper, width, grid_size in configurations:
        spectrum = phx.chemistry.UVVisibleSpectrumPlan(
            axis,
            phx.chemistry.SpectralLineShape.GAUSSIAN,
            lower,
            upper,
            fwhm=width,
            grid_size=grid_size,
            area_tolerance=1.0e-3,
        ).evaluate(manifold)
        assert bool(spectrum.successful)
        np.testing.assert_allclose(
            spectrum.integrated_strength,
            spectrum.expected_strength,
            rtol=1.0e-3,
        )


def test_nonresonant_raman_reports_activity_and_depolarization_separately():
    units = phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
    structure = phx.atomistic.AtomicStructure(
        [1, 1],
        [[-0.35, 0.0, 0.0], [0.35, 0.0, 0.0]],
        [1.0, 1.0],
        units.scale,
        particle_ids=[1, 2],
    )
    system = phx.atomistic.AtomisticSystemPlan.from_structure(
        structure, units, molecule_ids=[0, 0]
    )
    modes = np.zeros((2, 3, 1))
    modes[0, 0, 0] = -1.0 / np.sqrt(2.0)
    modes[1, 0, 0] = 1.0 / np.sqrt(2.0)
    vibration = phx.chemistry.VibrationalAnalysisResult(
        [1.0],
        [0.1],
        [1000.0],
        [False],
        modes,
        [0.5],
        external_mode_count=5,
        external_projection_residual=0.0,
        successful=True,
        stationary_point=phx.chemistry.StationaryPointKind.MINIMUM,
        units=units,
        source_system_id=system.system_id,
        source_geometry_id=structure.structure_id,
        plan_id="synthetic-vibration",
    )
    polarizability_unit = phx.units.derived_unit(
        "polarizability",
        (
            (units.charge_unit, 2),
            (units.scale.length_unit, 2),
            (units.scale.energy_unit, -1),
        ),
    )

    def polarizability(positions):
        coordinate = np.asarray(positions)
        bond = coordinate[1, 0] - coordinate[0, 0]
        tensor = np.diag([2.0 * bond, bond, 0.5 * bond])
        return phx.chemistry.StaticPolarizabilityResult(
            tensor,
            0.0,
            True,
            polarizability_unit,
            (f"bond-{bond:.12f}",),
        )

    provider = phx.chemistry.CallablePolarizabilityProvider(
        polarizability, "linear-polarizability", system.system_id
    )
    spectrum = phx.chemistry.RamanSpectrumPlan(
        system,
        provider,
        normal_coordinate_displacement=1.0e-4,
        grid_maximum=2000.0,
        grid_size=2001,
    ).evaluate(structure, vibration)

    assert bool(spectrum.successful)
    assert float(spectrum.activities[0]) > 0.0
    assert 0.0 <= float(spectrum.depolarization_ratios[0]) <= 0.75
    assert float(np.max(np.asarray(spectrum.broadened_intensity))) > 0.0
