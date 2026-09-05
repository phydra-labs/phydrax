import hashlib
import json
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.nucleic_acid_biophysics import NucleicAcidConstruct
from phydrax.applications.nucleic_acid_biophysics.observations import (
    AccessibilityReactivityModel,
    ChemicalMappingCondition,
    ChemicalMappingObservation,
    import_processed_rdat,
    IntervalDistanceReconstruction,
)
from phydrax.atomistic import AtomisticSystemPlan, AtomisticUnitSystem
from phydrax.qualification import ReferenceArtifactManifest
from phydrax.units import ANGSTROM


def manifest(payload=b"independent synthetic assay", *, training=True):
    return ReferenceArtifactManifest(
        "processed assay",
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        license_id="CC0-1.0",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=training,
        export_permitted=True,
        export_classification="unrestricted",
        nondimensionalization={"reactivity": 1.0},
        uncertainty=None,
        lineage_ids=("independently-authored-or-source-pinned",),
    )


def observation(
    values, *, replicate="r1", temperature=0.0, observed=None, lower=None, sd=None
):
    construct = NucleicAcidConstruct(("r",), ("ACG",), ("RNA",), (False,))
    return ChemicalMappingObservation(
        construct,
        construct.nucleotide_keys,
        values,
        np.ones(3) * 0.1 if sd is None else sd,
        reagent="synthetic-probe",
        condition=ChemicalMappingCondition(
            f"condition-{temperature}", (f"reduced-temperature:{temperature}",), 1.0
        ),
        replicate_id=replicate,
        preprocessing=("declared-background-subtraction",),
        source=manifest(),
        observed=observed,
        covariance_lower=lower,
    )


def test_observation_masks_negative_values_and_correlated_noise():
    lower = np.array([[0.1, 0.0], [0.06, 0.08]])
    obs = observation([-0.4, np.nan, 0.8], observed=[True, False, True], lower=lower)
    predicted = jnp.array([-0.3, 99.0, 0.7])
    score = eqx.filter_jit(obs.score)(predicted)
    # L^-1(data-prediction)=[-1,2], not the diagonal-noise quadratic 2.
    np.testing.assert_allclose(score.quadratic, 5.0, atol=1e-12)
    np.testing.assert_allclose(
        obs.score(predicted.at[1].set(-999.0)).log_probability, score.log_probability
    )
    with pytest.raises(ValueError):
        observation([1.0, 2.0, 3.0], sd=[0.1, 0.0, 0.1])
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError)):
        observation([1.0, 2.0, 3.0], lower=np.eye(3) * 0.2)


def test_shared_population_feature_fit_and_withheld_condition_prediction():
    signal = np.array([0.1, 0.5, 0.9])
    temperatures = np.array([0.0, 0.0, 1.0, 1.0])
    groups = ("r1", "r2", "r1", "r2")
    baselines = {"r1": -0.2, "r2": 0.4}
    observations = tuple(
        observation(
            baselines[group] + 1.5 * signal + 0.3 * t, replicate=group, temperature=t
        )
        for group, t in zip(groups, temperatures, strict=True)
    )
    model = AccessibilityReactivityModel(
        observations,
        (signal,) * 4,
        baseline_groups=groups,
        condition_features=temperatures[:, None],
        condition_names=("temperature",),
    )
    fitted = model.fit()
    assert bool(fitted.identifiable)
    for predicted, obs in zip(fitted.predictions, observations, strict=True):
        np.testing.assert_allclose(predicted, obs.reactivity, atol=1e-6)
    withheld = observation(baselines["r1"] + 1.5 * signal + 0.3 * 0.7, temperature=0.7)
    held_model = AccessibilityReactivityModel(
        (
            withheld,
            observation(
                baselines["r2"] + 1.5 * signal + 0.3 * 0.7,
                replicate="r2",
                temperature=0.7,
            ),
        ),
        (signal, signal),
        baseline_groups=("r1", "r2"),
        condition_features=[[0.7], [0.7]],
        condition_names=("temperature",),
    )
    np.testing.assert_allclose(
        held_model.predict(fitted.optimization.parameters)[0],
        withheld.reactivity,
        atol=1e-6,
    )
    constant = AccessibilityReactivityModel(
        (observations[0],),
        (np.ones(3) * 0.5,),
        baseline_groups=("r1",),
        condition_features=np.zeros((1, 0)),
    )
    assert not bool(constant.fit().identifiable)


def test_real_rdat_retains_mutant_constructs_negative_reactivity_and_scores():
    root = Path(__file__).resolve().parents[3] / "fixtures" / "nucleic_acid_biophysics"
    payload = (root / "TODEX_DMS_0000.rdat").read_bytes()
    record = json.loads((root / "TODEX_DMS_0000.source.json").read_text())
    source = ReferenceArtifactManifest(
        "RMDB:TODEX_DMS_0000",
        checksum_algorithm="sha256",
        checksum=record["sha256"],
        size_bytes=record["size_bytes"],
        license_id=record["license_id"],
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="CC0-database-content",
        nondimensionalization={"normalized-reactivity": 1.0},
        uncertainty=None,
        lineage_ids=(record["source_url"],),
    )
    imported = import_processed_rdat(
        payload, source, requested_use={}, error_semantics="standard-deviation"
    )
    assert len(imported.entries) == 16
    first, mutant = imported.entries[0], imported.entries[1]
    assert first.observation.construct.sequences != mutant.observation.construct.sequences
    assert first.observation.nucleotide_keys[0].position == 8
    assert first.observation.nucleotide_keys[-1].position == 142
    assert bool(jnp.any(first.observation.reactivity < 0))
    np.testing.assert_allclose(first.observation.reactivity[:3], [0.0267, 0.0016, 0.0418])
    np.testing.assert_allclose(
        first.observation.standard_deviation[:3], [0.0345, 0.0161, 0.0330]
    )
    prediction = jnp.zeros(first.observation.reactivity.shape)
    score = first.observation.score(prediction)
    np.testing.assert_allclose(
        score.quadratic,
        jnp.sum(
            (first.observation.reactivity / first.observation.standard_deviation) ** 2
        ),
        rtol=1e-12,
    )

    # Depositor design is a hypothesis only; withholding whole mutant constructs
    # tests the actual numerical prediction route without an accuracy claim.
    def feature(entry):
        return np.array(
            [
                float(entry.declared_structure[key.position] == ".")
                for key in entry.observation.nucleotide_keys
            ]
        )

    training = tuple(entry.observation for entry in imported.entries[:12])
    model = AccessibilityReactivityModel(
        training,
        tuple(feature(entry) for entry in imported.entries[:12]),
        baseline_groups=("shared",) * 12,
        condition_features=np.zeros((12, 0)),
    )
    fit = model.fit()
    withheld = imported.entries[12:]
    held_model = AccessibilityReactivityModel(
        tuple(entry.observation for entry in withheld),
        tuple(feature(entry) for entry in withheld),
        baseline_groups=("shared",) * 4,
        condition_features=np.zeros((4, 0)),
    )
    predicted = held_model.predict(fit.optimization.parameters)
    scores = tuple(
        entry.observation.score(y) for entry, y in zip(withheld, predicted, strict=True)
    )
    assert all(bool(score.successful) for score in scores)
    with pytest.raises(ValueError):
        import_processed_rdat(
            payload + b"\n",
            source,
            requested_use={},
            error_semantics="standard-deviation",
        )
    with pytest.raises(PermissionError):
        import_processed_rdat(
            payload,
            manifest(payload, training=False),
            requested_use={"training_use": True},
            error_semantics="standard-deviation",
        )


def test_native_interval_reconstruction_distinguishes_reflection():
    ids = [901, 77, 360, 29]
    units = AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
    system = AtomisticSystemPlan(
        ids, [6] * 4, [12.0] * 4, units, atom_type_ids=[0] * 4
    ).prepare()
    target = jnp.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    distance = np.array([1.0, np.sqrt(2), np.sqrt(2)])
    plan = IntervalDistanceReconstruction(
        system,
        [(901, 29), (77, 29), (360, 29)],
        distance - 0.01,
        distance + 0.01,
        np.full(3, 0.05),
        weights=np.ones(3),
        length_unit=ANGSTROM,
        sources=(manifest(),),
        requested_use={},
        chirality_atom_ids=[ids],
        chirality_sign=[1],
        minimum_volume=[0.05],
        chirality_standard_deviation=[0.01],
    )
    mirrored = target.at[:, 2].multiply(-1)
    np.testing.assert_allclose(
        plan.distances(mirrored), plan.distances(target), atol=1e-12
    )
    assert bool(plan.chirality(target).correct[0])
    assert not bool(plan.chirality(mirrored).correct[0])
    initial = target.at[3].set(jnp.array([0.15, -0.12, 0.65]))
    fixed = np.ones((4, 3), bool)
    fixed[3] = False
    result = plan.reconstruct(initial, fixed_mask=fixed, interval_tolerance=1e-5)
    assert bool(result.restraints_satisfied) and bool(result.chirality_qualified)
    np.testing.assert_allclose(result.positions[:3], target[:3], atol=0.0)
    np.testing.assert_allclose(result.initial_positions, initial, atol=0.0)
    with pytest.raises(ValueError):
        plan.reconstruct(initial, fixed_mask=np.ones((4, 3), bool))
