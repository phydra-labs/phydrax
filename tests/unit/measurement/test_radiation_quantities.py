import pytest

import phydrax as phx


_KIND_UNITS = {
    phx.measurement.RadiationQuantityKind.DEPOSITED_ENERGY: phx.units.JOULE,
    phx.measurement.RadiationQuantityKind.ABSORBED_DOSE: phx.units.GRAY,
    phx.measurement.RadiationQuantityKind.DOSE_TO_WATER: phx.units.GRAY,
    phx.measurement.RadiationQuantityKind.DOSE_TO_MEDIUM: phx.units.GRAY,
    phx.measurement.RadiationQuantityKind.KERMA: phx.units.GRAY,
    phx.measurement.RadiationQuantityKind.DOSE_RATE: phx.units.GRAY_PER_SECOND,
    phx.measurement.RadiationQuantityKind.RELATIVE_DOSE: phx.units.ONE,
    phx.measurement.RadiationQuantityKind.PARTICLE_FLUENCE: (
        phx.units.INVERSE_SQUARE_METER
    ),
    phx.measurement.RadiationQuantityKind.ENERGY_FLUENCE: (
        phx.units.JOULE_PER_SQUARE_METER
    ),
    phx.measurement.RadiationQuantityKind.LET: phx.units.JOULE_PER_METER,
    phx.measurement.RadiationQuantityKind.LINEAL_ENERGY: phx.units.JOULE_PER_METER,
    phx.measurement.RadiationQuantityKind.ACTIVITY: phx.units.BECQUEREL,
    phx.measurement.RadiationQuantityKind.ACTIVITY_CONCENTRATION: (
        phx.units.BECQUEREL_PER_CUBIC_METER
    ),
    phx.measurement.RadiationQuantityKind.TIME_INTEGRATED_ACTIVITY: (
        phx.units.BECQUEREL_SECOND
    ),
    phx.measurement.RadiationQuantityKind.TIME_INTEGRATED_ACTIVITY_CONCENTRATION: (
        phx.units.BECQUEREL_SECOND_PER_CUBIC_METER
    ),
}


def _quantity(kind, *, name=None, reference=None, support="voxel-cell-average"):
    return phx.measurement.resolve_radiation_quantity(
        kind.value if name is None else name,
        kind,
        _KIND_UNITS[kind],
        support_association=support,
        reference_configuration=(
            f"{kind.value}-defined-reference" if reference is None else reference
        ),
    )


def test_radiation_catalog_returns_standard_quantity_specs_for_supported_meanings():
    quantities = tuple(_quantity(kind) for kind in phx.measurement.RadiationQuantityKind)

    assert all(isinstance(value, phx.measurement.QuantitySpec) for value in quantities)
    assert {value.quantity_kind for value in quantities} == {
        kind.value for kind in phx.measurement.RadiationQuantityKind
    }
    assert all(value.namespace == "radiation" for value in quantities)


def test_radiation_meanings_remain_semantically_incompatible():
    absorbed = _quantity(phx.measurement.RadiationQuantityKind.ABSORBED_DOSE)
    water = _quantity(phx.measurement.RadiationQuantityKind.DOSE_TO_WATER)
    medium = _quantity(phx.measurement.RadiationQuantityKind.DOSE_TO_MEDIUM)
    kerma = _quantity(phx.measurement.RadiationQuantityKind.KERMA)
    let = _quantity(phx.measurement.RadiationQuantityKind.LET)
    lineal = _quantity(phx.measurement.RadiationQuantityKind.LINEAL_ENERGY)
    activity = _quantity(phx.measurement.RadiationQuantityKind.ACTIVITY)
    integrated_activity = _quantity(
        phx.measurement.RadiationQuantityKind.TIME_INTEGRATED_ACTIVITY
    )

    assert absorbed.unit.dimension == water.unit.dimension
    assert absorbed.unit.dimension == medium.unit.dimension
    assert absorbed.unit.dimension == kerma.unit.dimension
    assert not absorbed.compatible_with(water)
    assert not absorbed.compatible_with(medium)
    assert not absorbed.compatible_with(kerma)
    assert let.unit.dimension == lineal.unit.dimension
    assert not let.compatible_with(lineal)
    assert not activity.compatible_with(integrated_activity)


def test_reference_and_support_semantics_participate_in_compatibility():
    baseline = _quantity(
        phx.measurement.RadiationQuantityKind.DOSE_TO_MEDIUM,
        reference="water-equivalent-medium",
    )
    changed_medium = _quantity(
        phx.measurement.RadiationQuantityKind.DOSE_TO_MEDIUM,
        name="changed-medium-dose",
        reference="cortical-bone-medium",
    )
    changed_support = _quantity(
        phx.measurement.RadiationQuantityKind.DOSE_TO_MEDIUM,
        name="changed-support-dose",
        reference="water-equivalent-medium",
        support="point-sample",
    )

    assert not baseline.compatible_with(changed_medium)
    assert not baseline.compatible_with(changed_support)


def test_radiation_resolution_rejects_wrong_units_and_underspecified_meanings():
    with pytest.raises(ValueError, match="matching dimensions"):
        phx.measurement.resolve_radiation_quantity(
            "wrong-dose-unit",
            phx.measurement.RadiationQuantityKind.ABSORBED_DOSE,
            phx.units.JOULE,
            support_association="voxel-cell-average",
            reference_configuration="water-equivalent-medium",
        )
    with pytest.raises(ValueError, match="reference_configuration"):
        phx.measurement.resolve_radiation_quantity(
            "relative-dose",
            phx.measurement.RadiationQuantityKind.RELATIVE_DOSE,
            phx.units.ONE,
        )
    with pytest.raises(ValueError, match="support_association"):
        phx.measurement.resolve_radiation_quantity(
            "activity-concentration",
            phx.measurement.RadiationQuantityKind.ACTIVITY_CONCENTRATION,
            phx.units.BECQUEREL_PER_CUBIC_METER,
            reference_configuration="f18-source-at-acquisition-time",
        )


def test_nuclear_activity_resolution_uses_the_shared_radiation_identity():
    shared = _quantity(
        phx.measurement.RadiationQuantityKind.ACTIVITY,
        name="f18-activity",
        reference="f18-source-at-acquisition-time",
        support="source-region-total",
    )
    nuclear = phx.nuclear.resolve_nuclear_quantity(
        "f18-activity",
        phx.measurement.RadiationQuantityKind.ACTIVITY,
        phx.units.BECQUEREL,
        support_association="source-region-total",
        reference_configuration="f18-source-at-acquisition-time",
    )

    assert nuclear.namespace == "radiation"
    assert nuclear.quantity_id == shared.quantity_id
    assert nuclear.compatible_with(shared)
