from phydrax import applications
from phydrax.applications import skeletal_muscle
from phydrax.applications.skeletal_muscle import motor_units


def test_skeletal_muscle_facade_exports_only_owned_contracts():
    assert applications.skeletal_muscle is skeletal_muscle
    assert skeletal_muscle.__all__ == [
        "SKELETAL_MUSCLE_QUANTITIES",
        "SkeletalMuscleQuantitySpec",
        "cellular",
        "continuum",
        "electromyography",
        "energetics",
        "fatigue",
        "fibers",
        "interchange",
        "motor_units",
        "musculotendon",
        "personalization",
        "proprioception",
        "skeletal_muscle_quantity",
    ]
    for name in skeletal_muscle.__all__:
        assert name in vars(skeletal_muscle)
    assert len(skeletal_muscle.__all__) == len(set(skeletal_muscle.__all__))
    assert "DiscreteSystem" not in skeletal_muscle.__all__
    assert "CardiovascularQuantitySpec" not in skeletal_muscle.__all__


def test_motor_unit_facade_exports_each_concrete_symbol_once():
    expected = {
        "FuglevandForceVariabilityEvidence",
        "FuglevandWinterPatla1993Candidate",
        "FuglevandWinterPatla1993Evidence",
        "FuglevandWinterPatla1993Force",
        "FuglevandWinterPatla1993Parameters",
        "FuglevandWinterPatla1993Plan",
        "FuglevandWinterPatla1993QualificationEvidence",
        "FuglevandWinterPatla1993QualificationPlan",
        "FuglevandWinterPatla1993RandomInput",
        "FuglevandWinterPatla1993State",
        "FuglevandWinterPatla1993Status",
        "POTVIN_FUGLEVAND_2017_DOI",
        "POTVIN_FUGLEVAND_2017_MODEL_ID",
        "POTVIN_FUGLEVAND_2017_REFERENCE_SHA",
        "PotvinFuglevand2017Candidate",
        "PotvinFuglevand2017Evidence",
        "PotvinFuglevand2017Output",
        "PotvinFuglevand2017Parameters",
        "PotvinFuglevand2017Plan",
        "PotvinFuglevand2017State",
        "PotvinFuglevand2017Status",
        "PreparedPotvinFuglevand2017",
        "PreparedFuglevandWinterPatla1993",
        "potvin_fuglevand_2017_default_parameters",
        "commit_fuglevand_winter_patla_1993",
        "fuglevand_force_variability_evidence",
    }
    assert set(motor_units.__all__) == expected
    assert len(motor_units.__all__) == len(set(motor_units.__all__))
    for name in expected:
        if name.startswith("POTVIN_FUGLEVAND_2017_"):
            continue
        assert vars(motor_units)[name].__module__.startswith(
            "phydrax.applications.skeletal_muscle.motor_units"
        )
