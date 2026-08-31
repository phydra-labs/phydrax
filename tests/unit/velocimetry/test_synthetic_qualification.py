#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.velocimetry.imaging import DenseDisplacementField2D
from phydrax.velocimetry.synthetic import (
    generate_piv_case,
    PIVScenarioKind,
    PIVScenarioPlan,
    qualify_piv,
    qualify_ptv,
    qualify_stb,
    ScenarioSplitPolicy,
    split_synthetic_scenarios,
    SyntheticScenarioSplit,
)


def _split_cases():
    return tuple(
        generate_piv_case(
            PIVScenarioPlan(
                PIVScenarioKind.TRANSLATION,
                family_id=family,
                image_shape=(16, 16),
                particle_capacity=12,
                particle_density=0.02,
                seed=index,
            )
        )
        for index, family in enumerate(
            ("translation-a", "translation-a", "shear-b", "rotation-c", "dense-d")
        )
    )


def test_family_split_is_deterministic_disjoint_and_complete() -> None:
    cases = _split_cases()
    policy = ScenarioSplitPolicy(
        train_fraction=0.5,
        validation_fraction=0.25,
        test_fraction=0.25,
        seed=8,
    )

    first = split_synthetic_scenarios(cases, policy)
    second = split_synthetic_scenarios(cases, policy)

    assert first.split_id == second.split_id
    assert set(first.train_families).isdisjoint(first.validation_families)
    assert set(first.train_families).isdisjoint(first.test_families)
    assert set(first.validation_families).isdisjoint(first.test_families)
    index_partitions = (
        set(np.asarray(first.train_indices).tolist()),
        set(np.asarray(first.validation_indices).tolist()),
        set(np.asarray(first.test_indices).tolist()),
    )
    assert any({0, 1}.issubset(partition) for partition in index_partitions)
    combined = np.concatenate(
        (
            np.asarray(first.train_indices),
            np.asarray(first.validation_indices),
            np.asarray(first.test_indices),
        )
    )
    np.testing.assert_array_equal(np.sort(combined), np.arange(len(cases)))


def test_split_rejects_insufficient_families_duplicate_ids_and_overlap() -> None:
    cases = _split_cases()
    policy = ScenarioSplitPolicy()

    with pytest.raises(ValueError, match="at least three families"):
        split_synthetic_scenarios(cases[:2], policy)
    with pytest.raises(ValueError, match="unique"):
        split_synthetic_scenarios((cases[0], cases[0], cases[1], cases[2]), policy)
    with pytest.raises(ValueError, match="overlap"):
        SyntheticScenarioSplit(
            jnp.asarray((0, 1)),
            jnp.asarray((1,)),
            jnp.asarray((2,)),
            train_families=("a",),
            validation_families=("b",),
            test_families=("c",),
            scenario_ids=("0", "1", "2"),
            policy_id=policy.policy_id,
        )


def test_piv_qualification_exposes_bias_epe_and_coverage() -> None:
    case = generate_piv_case(
        PIVScenarioPlan(
            PIVScenarioKind.TRANSLATION,
            image_shape=(16, 18),
            particle_capacity=12,
            particle_density=0.02,
            displacement_rc=(1.0, -0.5),
        )
    )
    offset = jnp.asarray((0.5, -1.0))
    estimate = DenseDisplacementField2D(
        case.truth.positions_rc,
        case.truth.displacement_rc + offset,
        case.truth.valid,
        geometry_id=case.truth.geometry_id,
        field_id="biased-estimate",
    )

    result = qualify_piv(estimate, case.truth)

    np.testing.assert_allclose(result.bias_rc, offset)
    np.testing.assert_allclose(result.mean_endpoint_error, np.sqrt(1.25))
    np.testing.assert_allclose(result.root_mean_square_endpoint_error, np.sqrt(1.25))
    np.testing.assert_allclose(result.coverage, 1.0)
    assert {item.metric for item in result.evidence} == {
        "bias-rc",
        "coverage",
        "endpoint-error",
    }


def test_ptv_qualification_exposes_detection_triangulation_and_track_evidence() -> None:
    truth_xyz = jnp.asarray(
        (
            ((0.0, 0.0, 4.0), (0.2, 0.0, 4.0), (0.0, 0.0, 0.0)),
            ((0.1, 0.0, 4.0), (0.1, 0.0, 4.0), (0.0, 0.0, 0.0)),
        )
    )
    truth_valid = jnp.asarray(((True, True, False), (True, True, False)))
    reconstructed = truth_xyz.at[:, :2].add(jnp.asarray((0.01, -0.02, 0.03)))
    reconstructed_valid = truth_valid
    matches = jnp.asarray(((0, 1, -1), (0, 1, -1)))
    reconstructed_ids = jnp.asarray(((10, 99, -1), (10, 11, -1)))

    result = qualify_ptv(
        reconstructed,
        reconstructed_valid,
        matches,
        truth_xyz,
        truth_valid,
        reconstructed_track_ids=reconstructed_ids,
        truth_track_ids=jnp.asarray((10, 11, -1)),
        source_id="perfect-assignment",
    )

    np.testing.assert_allclose(result.detection_precision, 1.0)
    np.testing.assert_allclose(result.detection_recall, 1.0)
    np.testing.assert_allclose(result.triangulation_bias_xyz, (0.01, -0.02, 0.03))
    np.testing.assert_allclose(
        result.triangulation_root_mean_square_error,
        np.sqrt(0.01**2 + 0.02**2 + 0.03**2),
    )
    np.testing.assert_allclose(result.track_identity_accuracy, 0.75)
    np.testing.assert_allclose(result.track_completeness, 1.0)
    assert {item.metric for item in result.evidence} == {
        "detection",
        "track-completeness",
        "track-identity",
        "triangulation",
    }


def test_ptv_qualification_rejects_double_counted_truth_matches() -> None:
    truth = jnp.zeros((1, 2, 3))
    with pytest.raises(ValueError, match="unique"):
        qualify_ptv(
            truth,
            jnp.asarray(((True, True),)),
            jnp.asarray(((0, 0),)),
            truth,
            jnp.asarray(((True, True),)),
        )


def test_stb_qualification_exposes_masked_image_residual_evidence() -> None:
    observed = jnp.asarray(((1.0, 2.0), (3.0, 4.0)))
    reconstructed = observed + 1.0
    reconstructed = reconstructed.at[0, 0].set(jnp.nan)

    result = qualify_stb(reconstructed, observed, source_id="stb-residual")

    np.testing.assert_allclose(result.residual_bias, 1.0)
    np.testing.assert_allclose(result.residual_mean_absolute_error, 1.0)
    np.testing.assert_allclose(result.residual_root_mean_square_error, 1.0)
    np.testing.assert_allclose(result.coverage, 0.75)
    assert bool(result.finite)
    assert {item.metric for item in result.evidence} == {
        "coverage",
        "stb-image-residual",
    }
