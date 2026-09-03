#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import json

import pytest

from phydrax.lifecycle._resolved_run import (
    load_resolved_run_spec,
    resolve_run_spec,
    ResolvedRunSpec,
)
from phydrax.qualification._evidence import SupportDependency


def _resolved_run(
    scientific: tuple[SupportDependency, ...],
    deployment: tuple[SupportDependency, ...],
) -> ResolvedRunSpec:
    profile_ids = tuple(dependency.profile_id for dependency in scientific + deployment)
    return resolve_run_spec(
        scientific,
        deployment,
        release_index_id="release-2026-09",
        profile_ids=profile_ids,
        trust_policy_id="commercial-trust",
        valid_at=120,
        valid_from=100,
        valid_until=200,
        prepared_configuration_id="prepared-dns-run",
        precision_policy_id="precision-f64",
        resource_policy_id="resource-a100",
        checkpoint_policy_id="checkpoint-hourly",
        output_policy_id="output-commercial",
        repository_id="repository-primary",
        scheduler_id="scheduler-production",
        auth_policy_id="auth-workload",
    )


def test_resolved_run_has_exact_order_independent_identity() -> None:
    scientific = (
        SupportDependency("profile-equations", "tuple-equations"),
        SupportDependency("profile-closure", "tuple-closure"),
    )
    deployment = (
        SupportDependency("profile-runtime", "tuple-runtime"),
        SupportDependency("profile-storage", "tuple-storage"),
    )

    first = _resolved_run(scientific, deployment)
    reordered = _resolved_run(scientific[::-1], deployment[::-1])

    assert first.spec_id == reordered.spec_id
    assert first.to_json() == reordered.to_json()
    assert first.profile_ids == tuple(
        sorted(dependency.profile_id for dependency in scientific + deployment)
    )
    assert tuple(
        dependency.dependency_id for dependency in first.scientific_dependencies
    ) == tuple(sorted(dependency.dependency_id for dependency in scientific))


def test_resolved_run_strictly_reconstructs_canonical_json() -> None:
    value = _resolved_run(
        (SupportDependency("profile-science", "tuple-science"),),
        (SupportDependency("profile-deploy", "tuple-deploy"),),
    )

    restored = load_resolved_run_spec(value.to_json())

    assert restored.spec_id == value.spec_id
    assert restored.to_record() == value.to_record()
    assert restored.to_json() == value.to_json()


def test_resolved_run_rejects_unknown_missing_and_nonfinite_json() -> None:
    value = _resolved_run(
        (SupportDependency("profile-science", "tuple-science"),),
        (SupportDependency("profile-deploy", "tuple-deploy"),),
    )
    unknown = value.to_record()
    unknown["alias"] = "not-permitted"
    missing = value.to_record()
    del missing["scheduler_id"]
    nonfinite = value.to_record()
    nonfinite["valid_at"] = float("nan")

    with pytest.raises(ValueError, match="unknown fields"):
        ResolvedRunSpec.from_record(unknown)
    with pytest.raises(ValueError, match="missing fields"):
        ResolvedRunSpec.from_record(missing)
    with pytest.raises(ValueError, match="Non-finite"):
        load_resolved_run_spec(json.dumps(nonfinite))


def test_resolved_run_rejects_conflicting_tuple_dependencies() -> None:
    scientific = (SupportDependency("profile-shared", "tuple-a"),)
    conflicting = (SupportDependency("profile-shared", "tuple-b"),)
    duplicated = (SupportDependency("profile-shared", "tuple-a"),)

    with pytest.raises(ValueError, match="conflicting support-tuple"):
        _resolved_run(scientific, conflicting)
    with pytest.raises(ValueError, match="both scientific and deployment"):
        _resolved_run(scientific, duplicated)


def test_resolved_run_requires_exact_profiles_and_validity_window() -> None:
    dependency = SupportDependency("profile-science", "tuple-science")
    arguments = dict(
        release_index_id="release",
        trust_policy_id="trust",
        valid_at=20,
        valid_from=10,
        valid_until=30,
        prepared_configuration_id="prepared",
        precision_policy_id="precision",
        resource_policy_id="resource",
        checkpoint_policy_id="checkpoint",
        output_policy_id="output",
        repository_id="repository",
        scheduler_id="scheduler",
        auth_policy_id="auth",
    )

    with pytest.raises(ValueError, match="exactly match"):
        ResolvedRunSpec((dependency,), (), profile_ids=("profile-other",), **arguments)
    with pytest.raises(ValueError, match="within"):
        ResolvedRunSpec(
            (dependency,),
            (),
            profile_ids=("profile-science",),
            **{**arguments, "valid_at": 31},
        )
