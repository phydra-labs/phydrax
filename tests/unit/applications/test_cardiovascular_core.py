from dataclasses import FrozenInstanceError
from fractions import Fraction

import numpy as np
import pytest

from phydrax.applications.cardiovascular._case import (
    CARDIOVASCULAR_CASE_METADATA_KEYS,
    CardiovascularCaseManifest,
)
from phydrax.applications.cardiovascular._quantities import (
    CARDIOVASCULAR_QUANTITIES,
    cardiovascular_quantity,
    CardiovascularQuantitySpec,
)
from phydrax.units import LENGTH, MILLIVOLT, UnitDefinition, VOLT


EXPECTED_SI_FACTORS = {
    "time": Fraction(1, 1_000),
    "length": Fraction(1, 1_000),
    "area": Fraction(1, 1_000_000),
    "volume": Fraction(1, 1_000_000_000),
    "mass": Fraction(1, 1_000_000),
    "transmembrane_potential": Fraction(1, 1_000),
    "electric_field": Fraction(1),
    "membrane_current": Fraction(1, 1_000_000),
    "membrane_current_density": Fraction(1),
    "membrane_capacitance": Fraction(1, 1_000_000),
    "membrane_capacitance_density": Fraction(1),
    "electrical_conductivity": Fraction(1),
    "chemical_amount": Fraction(1, 1_000),
    "species_concentration": Fraction(1),
    "chemical_diffusivity": Fraction(1, 1_000),
    "concentration_rate": Fraction(1_000),
    "molar_surface_flux": Fraction(1_000_000),
    "pressure": Fraction(1_000),
    "velocity": Fraction(1),
    "acceleration": Fraction(1_000),
    "mass_density": Fraction(1_000),
    "force": Fraction(1, 1_000),
    "stress": Fraction(1_000),
    "strain": Fraction(1),
    "strain_rate": Fraction(1_000),
    "energy": Fraction(1, 1_000_000),
    "power": Fraction(1, 1_000),
    "dynamic_viscosity": Fraction(1),
    "volumetric_flow_rate": Fraction(1, 1_000_000),
    "hydraulic_resistance": Fraction(1_000_000_000),
    "hydraulic_inertance": Fraction(1_000_000),
    "hydraulic_compliance": Fraction(1, 1_000_000_000_000),
    "hydraulic_elastance": Fraction(1_000_000_000_000),
}


def _manifest(**overrides):
    values = {
        "case_id": "case-demo-001",
        "anatomy_id": "anatomy:sha256:001",
        "model_id": "model:sha256:002",
        "protocol_id": "protocol:sha256:003",
        "support_profile_id": "support:qualified-cpu",
        "release_id": "release:2026.09",
        "build_id": "build:sha256:004",
        "sbom_id": "sbom:sha256:005",
        "observation_ids": ("observation:ecg:006", "observation:pressure:007"),
        "license_ids": ("license:proprietary:008",),
        "data_rights_ids": ("rights:research:009",),
        "metadata": {
            "purpose": "research-use-only",
            "data_classification": "deidentified-aggregate",
        },
    }
    values.update(overrides)
    return CardiovascularCaseManifest(**values)


def test_all_kernel_quantities_have_exact_si_scales_and_round_trip():
    assert {
        name: spec.si_factor for name, spec in CARDIOVASCULAR_QUANTITIES.items()
    } == EXPECTED_SI_FACTORS
    assert len({spec.quantity_id for spec in CARDIOVASCULAR_QUANTITIES.values()}) == len(
        CARDIOVASCULAR_QUANTITIES
    )

    value = Fraction(37, 11)
    for name, spec in CARDIOVASCULAR_QUANTITIES.items():
        assert spec.name == name
        assert isinstance(spec.unit, UnitDefinition)
        assert spec.unit.symbol == spec.kernel_unit
        assert spec.from_si(spec.to_si(value)) == value
        assert spec.sign_convention
        assert spec.support_association
        assert spec.reference_configuration
        assert spec.spec_id == spec.quantity_id
        assert cardiovascular_quantity(name) is spec


def test_quantity_array_conversion_and_physical_metadata():
    pressure = cardiovascular_quantity("pressure")
    values = np.asarray([-2.5, 0.0, 13.25])
    np.testing.assert_array_equal(pressure.to_si(values), [-2500.0, 0.0, 13250.0])
    np.testing.assert_array_equal(pressure.from_si(pressure.to_si(values)), values)

    conductivity = cardiovascular_quantity("electrical_conductivity")
    assert conductivity.axes == ("component_i", "component_j")
    assert conductivity.quantity_kind == "electrical_conductivity"
    assert isinstance(conductivity.unit, UnitDefinition)
    assert conductivity.kernel_unit == "mS/mm"
    assert conductivity.si_unit == "S/m"


def test_quantity_identity_is_deterministic_and_metadata_sensitive():
    first = CardiovascularQuantitySpec(
        "paced_voltage",
        "electric_potential",
        MILLIVOLT,
        support_association="stimulus nodes",
        reference_configuration="extracellular potential",
    )
    second = CardiovascularQuantitySpec(
        "paced_voltage",
        "electric_potential",
        MILLIVOLT,
        support_association="stimulus nodes",
        reference_configuration="extracellular potential",
    )
    changed = CardiovascularQuantitySpec(
        "paced_voltage",
        "electric_potential",
        MILLIVOLT,
        sign_convention="positive depolarization",
        support_association="stimulus nodes",
        reference_configuration="extracellular potential",
    )
    changed_unit = CardiovascularQuantitySpec(
        "paced_voltage",
        "electric_potential",
        VOLT,
        support_association="stimulus nodes",
        reference_configuration="extracellular potential",
    )
    assert first == second
    assert first.quantity_id == second.quantity_id
    assert first.quantity_id != changed.quantity_id
    assert first.quantity_id != changed_unit.quantity_id
    with pytest.raises(FrozenInstanceError):
        first.name = "changed"


@pytest.mark.parametrize(
    ("arguments", "error"),
    [
        (("voltage", "electric_potential", "mV"), TypeError),
        (
            (
                "length",
                "length",
                UnitDefinition("other-m", LENGTH, "other-reference-system"),
            ),
            ValueError,
        ),
        (("pressure", "log_pressure", MILLIVOLT), ValueError),
        (("bad name", "electric_potential", MILLIVOLT), ValueError),
    ],
)
def test_quantity_specs_refuse_textual_ambiguous_or_reference_shifted_units(
    arguments, error
):
    with pytest.raises(error):
        CardiovascularQuantitySpec(*arguments)


def test_case_manifest_is_immutable_deterministic_and_order_canonical():
    first = _manifest()
    second = _manifest(
        observation_ids=tuple(reversed(first.observation_ids)),
        metadata=tuple(reversed(first.metadata)),
    )
    assert first == second
    assert first.manifest_id == second.manifest_id
    assert first.content_id == first.manifest_id
    assert first.observation_ids == tuple(sorted(first.observation_ids))
    assert dict(first.metadata_mapping) == dict(first.metadata)
    with pytest.raises(TypeError):
        first.metadata_mapping["purpose"] = "clinical"
    with pytest.raises(FrozenInstanceError):
        first.case_id = "changed"


def test_case_manifest_binds_every_declared_identity():
    baseline = _manifest()
    changes = {
        "case_id": "case-demo-002",
        "anatomy_id": "anatomy:sha256:101",
        "model_id": "model:sha256:102",
        "protocol_id": "protocol:sha256:103",
        "support_profile_id": "support:qualified-gpu",
        "release_id": "release:2026.10",
        "build_id": "build:sha256:104",
        "sbom_id": "sbom:sha256:105",
        "observation_ids": ("observation:ecg:106",),
        "license_ids": ("license:commercial:108",),
        "data_rights_ids": ("rights:validation:109",),
        "metadata": {"purpose": "qualification"},
    }
    for field_name, replacement in changes.items():
        assert _manifest(**{field_name: replacement}).manifest_id != baseline.manifest_id


def test_case_manifest_refuses_duplicate_or_malformed_identities():
    with pytest.raises(ValueError, match="duplicate"):
        _manifest(observation_ids=("observation:one", "observation:one"))
    with pytest.raises(ValueError, match="Every identity"):
        _manifest(license_ids=("build:sha256:004",))
    with pytest.raises(TypeError):
        _manifest(observation_ids="observation:one")
    with pytest.raises(ValueError, match="canonical technical identity"):
        _manifest(model_id=" model with spaces ")


@pytest.mark.parametrize(
    "overrides",
    [
        {"case_id": "patient-jane-doe"},
        {"metadata": {"patient_name": "jane"}},
        {"metadata": {"purpose": "patient-jane"}},
        {"metadata": {"purpose": "jane@example.org"}},
        {"metadata": {"purpose": "2020-01-02"}},
        {"metadata": {"purpose": "555-123-4567"}},
        {"metadata": {"unreviewed_note": "research"}},
    ],
)
def test_case_manifest_refuses_phi_linkable_or_non_allowlisted_metadata(overrides):
    with pytest.raises(ValueError):
        _manifest(**overrides)


def test_case_manifest_is_host_only_and_has_no_solver_or_schema_surface():
    manifest = _manifest(observation_ids=(), license_ids=(), data_rights_ids=())
    surface = set(dir(manifest))
    assert "solve" not in surface
    assert "state" not in surface
    assert "schema_version" not in surface
    assert "schema_version" not in manifest.__dataclass_fields__
    assert CARDIOVASCULAR_CASE_METADATA_KEYS == frozenset(
        {
            "cohort_definition",
            "consent_basis",
            "data_classification",
            "governance_policy",
            "intended_use",
            "jurisdiction",
            "pipeline",
            "purpose",
            "quality_policy",
            "retention_policy",
            "source_modality",
        }
    )
