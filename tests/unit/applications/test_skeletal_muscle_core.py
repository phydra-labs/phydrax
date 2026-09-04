from dataclasses import FrozenInstanceError
from fractions import Fraction

import pytest

from phydrax.applications.skeletal_muscle import (
    SKELETAL_MUSCLE_QUANTITIES,
    skeletal_muscle_quantity,
    SkeletalMuscleQuantitySpec,
)


EXPECTED_QUANTITIES = {
    "brain_effort_rate",
    "common_excitation",
    "contraction_time",
    "force_capacity_fraction",
    "force_standard_uncertainty",
    "gamma_drive_frequency",
    "independent_excitation",
    "mechanical_power",
    "motor_unit_compartment_fraction",
    "motor_unit_event_time",
    "motor_unit_firing_rate",
    "muscle_activation",
    "muscle_fiber_force",
    "muscle_fiber_length",
    "muscle_fiber_velocity",
    "muscle_metabolic_energy",
    "muscle_metabolic_power",
    "musculotendon_length",
    "musculotendon_velocity",
    "normalized_fascicle_length",
    "normalized_muscle_fiber_length",
    "normalized_muscle_fiber_velocity",
    "normalized_tendon_force",
    "normalized_tendon_length",
    "observed_force",
    "passive_fiber_elastic_energy",
    "pennation_angle",
    "physical_force_scale",
    "raw_provider_force",
    "recruitment_duration",
    "relative_isometric_force",
    "relative_muscle_force",
    "relative_twitch_force",
    "skeletal_continuum_stiffness",
    "skeletal_muscle_cytosolic_calcium_concentration",
    "skeletal_muscle_force_bearing_crossbridge_concentration",
    "skeletal_muscle_membrane_current_density",
    "skeletal_muscle_sr_calcium_concentration",
    "skeletal_muscle_stimulus_current_density",
    "skeletal_peak_active_nominal_stress",
    "skeletal_prescribed_activation",
    "spindle_afferent_rate",
    "surface_electric_potential",
    "tendon_elastic_energy",
    "tendon_force",
    "tendon_length",
    "tendon_velocity",
    "time",
}


def test_skeletal_quantities_are_exact_complete_and_immutable():
    assert EXPECTED_QUANTITIES == set(SKELETAL_MUSCLE_QUANTITIES)
    assert len({value.quantity_id for value in SKELETAL_MUSCLE_QUANTITIES.values()}) == len(
        SKELETAL_MUSCLE_QUANTITIES
    )
    for name, quantity in SKELETAL_MUSCLE_QUANTITIES.items():
        assert quantity.name == name
        assert quantity.si_factor > 0
        assert quantity.sign_convention
        assert quantity.support_association
        assert quantity.reference_configuration
        assert quantity.spec_id == quantity.quantity_id
        assert skeletal_muscle_quantity(name) is quantity
        assert quantity.from_si(quantity.to_si(Fraction(13, 7))) == Fraction(13, 7)

    with pytest.raises(FrozenInstanceError):
        skeletal_muscle_quantity("time").name = "changed"
    with pytest.raises(KeyError):
        skeletal_muscle_quantity("unknown")
    assert (
        skeletal_muscle_quantity(
            "skeletal_muscle_stimulus_current_density"
        ).to_si(1.0)
        == 0.01
    )
    assert (
        skeletal_muscle_quantity(
            "skeletal_muscle_cytosolic_calcium_concentration"
        ).to_si(1.0)
        == 0.001
    )
    assert skeletal_muscle_quantity("motor_unit_event_time").to_si(1.0) == 0.001
    assert skeletal_muscle_quantity("spindle_afferent_rate").to_si(1.0) == 1.0


def test_skeletal_quantity_identity_is_domain_and_metadata_sensitive():
    baseline = SkeletalMuscleQuantitySpec(
        "drive",
        "dimensionless",
        "1",
        "1",
        Fraction(1),
        sign_convention="nonnegative",
        support_association="motor-unit population",
        reference_configuration="named model scale",
    )
    equivalent = SkeletalMuscleQuantitySpec(
        "drive",
        "dimensionless",
        "1",
        "1",
        1.0,
        sign_convention="nonnegative",
        support_association="motor-unit population",
        reference_configuration="named model scale",
    )
    changed = SkeletalMuscleQuantitySpec(
        "drive",
        "dimensionless",
        "1",
        "1",
        1,
        sign_convention="signed",
        support_association="motor-unit population",
        reference_configuration="named model scale",
    )
    assert baseline == equivalent
    assert baseline.quantity_id == equivalent.quantity_id
    assert baseline.quantity_id != changed.quantity_id

    with pytest.raises(ValueError):
        SkeletalMuscleQuantitySpec(
            "drive",
            "dimensionless",
            "percent",
            "1",
            Fraction(1, 100),
        )
