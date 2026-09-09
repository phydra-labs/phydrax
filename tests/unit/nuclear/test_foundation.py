from __future__ import annotations

import hashlib

import numpy as np
import pytest

import phydrax as phx


def _reference(payload: bytes, name: str = "fixture"):
    return phx.qualification.ReferenceArtifactManifest(
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


def test_nuclide_identity_is_independent_of_evaluated_mass_data():
    deuterium = phx.nuclear.NuclideKey(1, 2)
    species = (phx.nuclear.NuclearSpeciesKey.from_nuclide(deuterium),)
    first = phx.nuclear.NuclearSpeciesTable(
        species, np.asarray([3.3435837724e-27]), _reference(b"a", "mass-a")
    )
    second = phx.nuclear.NuclearSpeciesTable(
        species, np.asarray([3.343583719e-27]), _reference(b"b", "mass-b")
    )

    assert first.species[0].species_id == second.species[0].species_id
    assert first.table_id != second.table_id
    assert int(first.prepare().baryon_numbers[0]) == 2
    assert int(first.prepare().charge_numbers[0]) == 1


def test_nuclear_data_provenance_binds_processing_and_rights():
    reference = _reference(b"evaluated")
    raw = phx.nuclear.NuclearDataProvenance(
        reference,
        "https://example.invalid/evaluated",
        "fixture-library",
        "release-a",
        "evaluation-a",
    )
    processed = phx.nuclear.NuclearDataProvenance(
        reference,
        "https://example.invalid/evaluated",
        "fixture-library",
        "release-a",
        "evaluation-a",
        processing_tool="fixture-processor",
        processing_release="build-a",
        processing_parameters={"temperature_k": 900.0},
        parent_data_ids=(raw.provenance_id,),
    )

    assert not raw.is_processed
    assert processed.is_processed
    assert processed.provenance_id != raw.provenance_id
    with pytest.raises(ValueError, match="processing tool"):
        phx.nuclear.NuclearDataProvenance(
            reference,
            "https://example.invalid/evaluated",
            "fixture-library",
            "release-a",
            "evaluation-a",
            processing_parameters={"temperature_k": 900.0},
        )


def test_energy_groups_and_composition_preserve_physical_semantics():
    groups = phx.nuclear.EnergyGroupStructure(
        np.asarray([0.0, 1.0, 14.0]),
        phx.units.MEGAELECTRONVOLT,
        source_id="synthetic-groups",
    )
    np.testing.assert_allclose(
        groups.edges_in(phx.units.MEGAELECTRONVOLT), [0.0, 1.0, 14.0]
    )
    location = groups.prepare().locate(
        np.asarray([-1.0, 0.0, 1.0, 14.0, 15.0])
        * float(phx.units.conversion_factor(phx.units.MEGAELECTRONVOLT, phx.units.JOULE))
    )
    np.testing.assert_array_equal(location.valid, [False, True, True, True, False])
    np.testing.assert_array_equal(location.indices, [0, 0, 1, 1, 1])

    hydrogen = phx.nuclear.NuclideKey(1, 1)
    deuterium = phx.nuclear.NuclideKey(1, 2)
    nuclides = (hydrogen, deuterium)
    species = tuple(phx.nuclear.NuclearSpeciesKey.from_nuclide(v) for v in nuclides)
    table = phx.nuclear.NuclearSpeciesTable(
        species,
        np.asarray([1.6735575e-27, 3.3435838e-27]),
        _reference(b"masses", "mass-table"),
    )
    atoms = phx.nuclear.NuclideComposition(
        nuclides,
        np.asarray([0.75, 0.25]),
        phx.nuclear.CompositionBasis.ATOM_FRACTION,
        "synthetic-composition",
    )
    mass = atoms.convert_basis(phx.nuclear.CompositionBasis.MASS_FRACTION, table)
    restored = mass.composition.convert_basis(
        phx.nuclear.CompositionBasis.ATOM_FRACTION, table
    )
    np.testing.assert_allclose(restored.composition.fractions, atoms.fractions)
    assert mass.closure_residual < 1.0e-15
    assert restored.closure_residual < 1.0e-15


def test_multigroup_source_uses_shared_measurement_contracts():
    groups = phx.nuclear.EnergyGroupStructure(
        [0.0, 1.0, 2.0], phx.units.MEGAELECTRONVOLT, source_id="groups"
    )
    support = phx.measurement.IndexSampleSupport(
        (2, 2), ("region", "energy_group"), frame_id="fixture-regions"
    )
    unit = phx.units.derived_unit(
        "fixture-source-density",
        ((phx.units.METER, -3), (phx.units.SECOND, -1)),
    )
    quantity = phx.nuclear.resolve_nuclear_quantity(
        "neutron-source",
        "particle_source_density",
        unit,
        axes=("region", "energy_group"),
        support_association="cell",
    )
    field = phx.measurement.QuantityField(
        "neutron-source-field",
        quantity,
        phx.measurement.ValueLayout.scalar(),
        support,
        phx.measurement.SamplingSemantics(
            phx.measurement.SpatialSamplingKind.CELL_AVERAGE
        ),
        np.asarray([[1.0, 2.0], [3.0, 4.0]]),
    )
    neutron = phx.nuclear.NuclearSpeciesKey.from_particle(
        phx.nuclear.NuclearParticleKind.NEUTRON
    )
    source = phx.nuclear.MultigroupParticleSource(field, groups, neutron)

    prepared = source.prepare()
    assert prepared.field.values.shape == (2, 2)
    assert prepared.energy_groups.group_count == 2
    assert source.source_id == prepared.source_id


def test_group_reaction_rate_contracts_the_energy_axis():
    cross_section = np.asarray([[1.0e-28, 2.0e-28], [3.0e-28, 4.0e-28]])
    flux = np.asarray([[1.0e18, 2.0e18], [3.0e18, 4.0e18]])
    expected = np.sum(cross_section * flux, axis=-1)
    np.testing.assert_allclose(
        phx.nuclear.group_reaction_rate(cross_section, flux), expected
    )
