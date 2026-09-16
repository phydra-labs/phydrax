#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib

import numpy as np
import pytest

import phydrax as phx


def _provenance(name: str) -> phx.nuclear.NuclearDataProvenance:
    payload = name.encode()
    reference = phx.qualification.ReferenceArtifactManifest(
        name,
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        license_id="fixture-license",
        commercial_use_permitted=False,
        redistribution_permitted=False,
        training_use_permitted=False,
        export_permitted=False,
        export_classification="fixture",
        nondimensionalization={"identity": 1.0},
        uncertainty=None,
        lineage_ids=("synthetic",),
    )
    return phx.nuclear.NuclearDataProvenance(
        reference,
        f"https://example.invalid/{name}",
        "synthetic-diagnostic-photon-data",
        "release-a",
        name,
    )


def _coefficient_unit():
    return phx.units.derived_unit(
        "m2/kg",
        ((phx.units.METER, 2), (phx.units.KILOGRAM, -1)),
    )


def _table(
    *,
    provenance: phx.nuclear.NuclearDataProvenance | None = None,
    interpolation=phx.equations.DiagnosticPhotonInterpolationPolicy.LINEAR,
):
    grid = phx.equations.PhotonEnergyGrid(np.asarray([1.0e-16, 2.0e-16, 4.0e-16]))
    return phx.equations.DiagnosticPhotonCoefficientTable(
        phx.equations.DiagnosticPhotonCoefficientRole.MASS_ATTENUATION,
        grid,
        ("water", "bone"),
        np.asarray([[1.0, 3.0, 7.0], [2.0, 4.0, 8.0]]),
        _coefficient_unit(),
        _provenance("table") if provenance is None else provenance,
        interpolation,
    )


def test_photon_energy_grid_requires_positive_strictly_increasing_joules():
    grid = phx.equations.PhotonEnergyGrid(np.asarray([1.0e-16, 2.0e-16, 4.0e-16]))

    np.testing.assert_array_equal(
        grid.energy_j,
        np.asarray([1.0e-16, 2.0e-16, 4.0e-16]),
    )
    with pytest.raises(ValueError, match="positive"):
        phx.equations.PhotonEnergyGrid(np.asarray([0.0, 1.0e-16]))
    with pytest.raises(ValueError, match="strictly increasing"):
        phx.equations.PhotonEnergyGrid(np.asarray([2.0e-16, 1.0e-16]))


def test_diagnostic_coefficient_table_rejects_invalid_values_and_units():
    grid = phx.equations.PhotonEnergyGrid(np.asarray([1.0e-16, 2.0e-16]))
    provenance = _provenance("invalid-table")
    arguments = (
        phx.equations.DiagnosticPhotonCoefficientRole.MASS_ATTENUATION,
        grid,
        ("water",),
    )

    with pytest.raises(ValueError, match="finite and nonnegative"):
        phx.equations.DiagnosticPhotonCoefficientTable(
            *arguments,
            np.asarray([[1.0, -1.0]]),
            _coefficient_unit(),
            provenance,
            phx.equations.DiagnosticPhotonInterpolationPolicy.LINEAR,
        )
    with pytest.raises(ValueError, match="finite and nonnegative"):
        phx.equations.DiagnosticPhotonCoefficientTable(
            *arguments,
            np.asarray([[1.0, np.nan]]),
            _coefficient_unit(),
            provenance,
            phx.equations.DiagnosticPhotonInterpolationPolicy.LINEAR,
        )
    with pytest.raises(ValueError, match="area-per-mass"):
        phx.equations.DiagnosticPhotonCoefficientTable(
            *arguments,
            np.asarray([[1.0, 2.0]]),
            phx.units.METER,
            provenance,
            phx.equations.DiagnosticPhotonInterpolationPolicy.LINEAR,
        )
    with pytest.raises(ValueError, match="strictly positive"):
        phx.equations.DiagnosticPhotonCoefficientTable(
            *arguments,
            np.asarray([[0.0, 2.0]]),
            _coefficient_unit(),
            provenance,
            phx.equations.DiagnosticPhotonInterpolationPolicy.LOG_LOG,
        )


def test_ordered_material_evaluation_interpolates_without_extrapolation():
    table = _table()
    evaluation = table.evaluate(
        np.asarray([1.5e-16, 2.0e-16, 5.0e-16]),
        ("water", "bone"),
    )

    np.testing.assert_allclose(
        evaluation.coefficient,
        np.asarray([[2.0, 3.0, 0.0], [3.0, 4.0, 0.0]]),
    )
    np.testing.assert_array_equal(evaluation.evidence.supported, [True, True, False])
    np.testing.assert_array_equal(evaluation.evidence.interpolated, [True, False, False])
    assert evaluation.unit == table.unit
    assert evaluation.provenance_id == table.provenance.provenance_id
    assert evaluation.evidence.provenance_id == table.provenance.provenance_id

    with pytest.raises(ValueError, match="ordered basis"):
        table.evaluate(np.asarray([2.0e-16]), ("bone", "water"))


def test_log_log_interpolation_uses_logarithmic_energy_and_value_coordinates():
    grid = phx.equations.PhotonEnergyGrid(np.asarray([1.0, 4.0]))
    table = phx.equations.DiagnosticPhotonCoefficientTable(
        phx.equations.DiagnosticPhotonCoefficientRole.MASS_ENERGY_ABSORPTION,
        grid,
        ("water",),
        np.asarray([[1.0, 16.0]]),
        _coefficient_unit(),
        _provenance("log-log"),
        phx.equations.DiagnosticPhotonInterpolationPolicy.LOG_LOG,
    )

    evaluation = table.evaluate(np.asarray([2.0]), ("water",))

    np.testing.assert_allclose(evaluation.coefficient, np.asarray([[4.0]]))
    np.testing.assert_allclose(evaluation.evidence.upper_weights, np.asarray([0.5]))


def test_evaluation_refuses_an_explicit_mismatched_provenance_pin():
    table = _table(provenance=_provenance("authoritative"))

    with pytest.raises(ValueError, match="provenance"):
        table.evaluate(
            np.asarray([2.0e-16]),
            ("water", "bone"),
            expected_provenance=_provenance("other"),
        )


def test_diagnostic_coefficients_are_separate_from_thermal_radiation_means():
    table = _table()
    evaluation = table.evaluate(np.asarray([2.0e-16]), ("water", "bone"))

    assert not isinstance(table, phx.equations.RadiationCoefficientTable)
    with pytest.raises(AttributeError):
        phx.equations.radiation_means(
            np.asarray(300.0),
            evaluation,
            evaluation,
            table.energy_grid,
        )
