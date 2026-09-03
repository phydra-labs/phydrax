#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from phydrax.lifecycle._resolved_run import ResolvedRunSpec
from phydrax.qualification._evidence import QualificationEvidence, SupportDependency
from tools.commercial_platform_qualification import (
    assemble_commercial_platform_candidate,
    FaultObservation,
    GATES,
    produce_provider_qualification,
    QualificationContext,
    required_boundaries,
    ROUTE_BOUNDARIES,
    standard_fault_matrix,
    verify_commercial_platform_candidate,
    verify_provider_qualification,
)


_REPOSITORY_ROUTES = frozenset(
    (
        "posix-repository",
        "s3-repository",
        "hpc-filesystem",
        "chunk-lifecycle",
        "production-repository",
    )
)
_SCHEDULER_ROUTES = frozenset(("slurm", "kubernetes"))
_AUTHENTICATION_ROUTES = frozenset(("oidc-jwks", "mtls"))


class _InjectedProvider:
    def __init__(
        self,
        provider_id: str,
        deployment_id: str,
        /,
        *,
        wrong_boundary: str | None = None,
        leak_boundary: str | None = None,
        hidden_effect_boundary: str | None = None,
    ):
        self.provider_id = provider_id
        self.deployment_id = deployment_id
        self.wrong_boundary = wrong_boundary
        self.leak_boundary = leak_boundary
        self.hidden_effect_boundary = hidden_effect_boundary
        self.effects: list[str] = []
        self.calls: list[str] = []

    def effect_log(self):
        return tuple(self.effects)

    def exercise(self, case, /):
        self.calls.append(case.boundary_id)
        effects = list(case.allowed_effects)
        if case.boundary_id == self.hidden_effect_boundary:
            effects.append(f"undeclared:{case.boundary_id}")
        self.effects.extend(effects)
        facts = dict(case.expected_facts)
        if case.boundary_id == self.wrong_boundary:
            facts["contract_satisfied"] = False
        if case.boundary_id == self.leak_boundary:
            facts["authorization"] = "Bearer qualification-super-secret"
        return FaultObservation(facts, declared_effects=effects)


def _bindings(route: str, /, *, deployment_id: str | None = None):
    provider_id = f"provider.{route}"
    deployment = f"deployment.{route}" if deployment_id is None else deployment_id
    dependency = SupportDependency(f"profile.{route}", f"tuple.{route}")
    repository_id = provider_id if route in _REPOSITORY_ROUTES else "repository.bound"
    scheduler_id = provider_id if route in _SCHEDULER_ROUTES else "scheduler.bound"
    auth_policy_id = (
        provider_id if route in _AUTHENTICATION_ROUTES else "authentication.bound"
    )
    resolved = ResolvedRunSpec(
        (),
        (dependency,),
        release_index_id="release.candidate",
        profile_ids=(dependency.profile_id,),
        trust_policy_id="trust.candidate",
        valid_at=20,
        valid_from=1,
        valid_until=100,
        prepared_configuration_id="configuration.candidate",
        precision_policy_id="precision.float64",
        resource_policy_id="resources.qualification",
        checkpoint_policy_id="checkpoint.qualification",
        output_policy_id="output.qualification",
        repository_id=repository_id,
        scheduler_id=scheduler_id,
        auth_policy_id=auth_policy_id,
    )
    context = QualificationContext(
        build_id="build.qualification",
        environment_id=deployment,
        backend="injected",
        topology="single-process",
        precision="float64",
        reduction="deterministic",
        replay_id=f"replay.{route}",
        reviewer_id="reviewer.qualification",
        issued_at=10,
        expires_at=90,
        evaluated_at=20,
    )
    source = tuple(
        QualificationEvidence(
            kind,
            "passed",
            (resolved.spec_id, dependency.dependency_id),
            build_id=context.build_id,
            environment_id=context.environment_id,
            backend=context.backend,
            topology=context.topology,
            precision=context.precision,
            reduction=context.reduction,
            replay_id=context.replay_id,
            criteria_ids=(criterion,),
            raw_artifact_ids=(f"raw.{criterion}",),
            reviewer_id=context.reviewer_id,
            issued_at=10,
            expires_at=90,
            reason="Injected exact source evidence passed.",
        )
        for kind, criterion in (
            ("scientific", "scientific.reference"),
            ("performance", "performance.reference"),
        )
    )
    matrix = standard_fault_matrix(
        route,
        provider_id,
        deployment,
        dependency.dependency_id,
    )
    return provider_id, deployment, dependency, resolved, context, source, matrix


def _produce(route: str, provider: _InjectedProvider | None = None):
    provider_id, deployment, dependency, resolved, context, source, matrix = _bindings(
        route
    )
    injected = (
        _InjectedProvider(provider_id, deployment) if provider is None else provider
    )
    artifact = produce_provider_qualification(
        injected,
        matrix,
        context=context,
        support_dependencies=(dependency,),
        resolved_run_spec=resolved,
        source_evidence=source,
        scientific_criteria=("scientific.reference",),
        performance_criteria=("performance.reference",),
    )
    return artifact, injected


def test_all_platform_fault_boundaries_are_exercised_and_assembled():
    artifacts = []
    exercised = {}
    for route in ROUTE_BOUNDARIES:
        artifact, provider = _produce(route)
        verify_provider_qualification(artifact)
        artifacts.append(artifact)
        exercised[route] = tuple(sorted(provider.calls))
        assert artifact["status"] == "passed"
        assert tuple(gate["gate"] for gate in artifact["gates"]) == GATES
        assert all(gate["outcome"] == "passed" for gate in artifact["gates"])
        assert artifact["signed"] is False
        assert artifact["release_ready"] is False
        assert "schema_version" not in artifact
        assert exercised[route] == tuple(sorted(required_boundaries(route)))

    candidate = assemble_commercial_platform_candidate(artifacts)
    verify_commercial_platform_candidate(candidate)
    assert candidate["status"] == "passed"
    assert candidate["signed"] is False
    assert candidate["release_ready"] is False
    assert set(candidate["required_routes"]) == set(ROUTE_BOUNDARIES)


@pytest.mark.parametrize(
    ("route", "boundary"),
    (
        ("chunk-lifecycle", "chunk.duplicate-write"),
        ("chunk-lifecycle", "chunk.stale-attempt"),
        ("durable-service", "service.cross-tenant-denial"),
        ("durable-service", "service.stale-execution-attempt"),
        ("durable-service", "service.duplicate-delivery"),
        ("slurm", "slurm.duplicate-submit"),
        ("slurm", "slurm.stale-attempt"),
        ("kubernetes", "kubernetes.resource-version-conflict"),
        ("oidc-jwks", "oidc.expiry"),
        ("oidc-jwks", "jwks.key-revocation"),
        ("mtls", "mtls.expiry"),
        ("mtls", "mtls.revocation"),
        ("ed25519", "ed25519.key-expiry"),
        ("ed25519", "ed25519.key-revocation"),
        ("kms", "kms.key-expiry"),
        ("kms", "kms.key-revocation"),
        ("configuration-migration", "migration.lossy-denial"),
        ("configuration-migration", "migration.ambiguous-path"),
        ("configuration-migration", "migration.immutable-parent-rollback"),
    ),
)
def test_fault_contract_failures_remain_boundary_and_gate_specific(route, boundary):
    provider_id, deployment, dependency, resolved, context, source, matrix = _bindings(
        route
    )
    provider = _InjectedProvider(provider_id, deployment, wrong_boundary=boundary)
    artifact = produce_provider_qualification(
        provider,
        matrix,
        context=context,
        support_dependencies=(dependency,),
        resolved_run_spec=resolved,
        source_evidence=source,
        scientific_criteria=("scientific.reference",),
        performance_criteria=("performance.reference",),
    )

    failed = [
        observation
        for observation in artifact["observations"]
        if observation["outcome"] == "failed"
    ]
    assert [observation["boundary_id"] for observation in failed] == [boundary]
    assert artifact["status"] == "failed"
    assert artifact["gates"][0]["outcome"] == "passed"
    assert artifact["gates"][1]["outcome"] == "passed"


def test_secret_leakage_is_failed_and_removed_from_candidate_records():
    route = "redaction-support"
    boundary = "redaction.recursive"
    provider_id, deployment, dependency, resolved, context, source, matrix = _bindings(
        route
    )
    provider = _InjectedProvider(provider_id, deployment, leak_boundary=boundary)
    artifact = produce_provider_qualification(
        provider,
        matrix,
        context=context,
        support_dependencies=(dependency,),
        resolved_run_spec=resolved,
        source_evidence=source,
        scientific_criteria=("scientific.reference",),
        performance_criteria=("performance.reference",),
        forbidden_values=("qualification-super-secret",),
    )

    payload = json.dumps(artifact, sort_keys=True)
    leaked = next(
        observation
        for observation in artifact["observations"]
        if observation["boundary_id"] == boundary
    )
    assert leaked["secret_leak_detected"] is True
    assert leaked["outcome"] == "failed"
    assert "qualification-super-secret" not in payload
    assert "Bearer qualification" not in payload
    verify_provider_qualification(artifact)


def test_undeclared_effects_fail_and_provider_construction_is_inert():
    route = "oidc-jwks"
    boundary = "platform.no-hidden-external-effects"
    provider_id, deployment, dependency, resolved, context, source, matrix = _bindings(
        route
    )
    provider = _InjectedProvider(
        provider_id,
        deployment,
        hidden_effect_boundary=boundary,
    )
    assert provider.calls == []
    assert provider.effect_log() == ()

    artifact = produce_provider_qualification(
        provider,
        matrix,
        context=context,
        support_dependencies=(dependency,),
        resolved_run_spec=resolved,
        source_evidence=source,
        scientific_criteria=("scientific.reference",),
        performance_criteria=("performance.reference",),
    )

    hidden = next(
        observation
        for observation in artifact["observations"]
        if observation["boundary_id"] == boundary
    )
    assert hidden["effects_matched"] is False
    assert hidden["outcome"] == "failed"
    assert artifact["status"] == "failed"

    constructor_effect = _InjectedProvider(provider_id, deployment)
    constructor_effect.effects.append("undeclared:constructor-network")
    constructor_artifact = produce_provider_qualification(
        constructor_effect,
        matrix,
        context=context,
        support_dependencies=(dependency,),
        resolved_run_spec=resolved,
        source_evidence=source,
        scientific_criteria=("scientific.reference",),
        performance_criteria=("performance.reference",),
    )
    constructor_boundary = next(
        observation
        for observation in constructor_artifact["observations"]
        if observation["boundary_id"] == boundary
    )
    assert constructor_boundary["initial_effects_clean"] is False
    assert constructor_boundary["outcome"] == "failed"
    verify_provider_qualification(constructor_artifact)


def test_provider_deployment_and_support_mismatch_fail_before_exercise():
    route = "s3-repository"
    provider_id, deployment, dependency, resolved, context, source, matrix = _bindings(
        route
    )
    wrong_provider = _InjectedProvider("provider.other", deployment)
    with pytest.raises(ValueError, match="provider identity"):
        produce_provider_qualification(
            wrong_provider,
            matrix,
            context=context,
            support_dependencies=(dependency,),
            resolved_run_spec=resolved,
            source_evidence=source,
            scientific_criteria=("scientific.reference",),
            performance_criteria=("performance.reference",),
        )
    assert wrong_provider.calls == []

    foreign_matrix = standard_fault_matrix(
        route,
        "provider.other",
        deployment,
        dependency.dependency_id,
    )
    foreign_provider = _InjectedProvider("provider.other", deployment)
    with pytest.raises(ValueError, match="resolved repository"):
        produce_provider_qualification(
            foreign_provider,
            foreign_matrix,
            context=context,
            support_dependencies=(dependency,),
            resolved_run_spec=resolved,
            source_evidence=source,
            scientific_criteria=("scientific.reference",),
            performance_criteria=("performance.reference",),
        )
    assert foreign_provider.calls == []

    wrong_deployment = _InjectedProvider(provider_id, "deployment.other")
    with pytest.raises(ValueError, match="deployment identity"):
        produce_provider_qualification(
            wrong_deployment,
            matrix,
            context=context,
            support_dependencies=(dependency,),
            resolved_run_spec=resolved,
            source_evidence=source,
            scientific_criteria=("scientific.reference",),
            performance_criteria=("performance.reference",),
        )
    assert wrong_deployment.calls == []

    foreign_dependency = SupportDependency("profile.foreign", "tuple.foreign")
    provider = _InjectedProvider(provider_id, deployment)
    with pytest.raises(ValueError, match="exactly equal"):
        produce_provider_qualification(
            provider,
            matrix,
            context=context,
            support_dependencies=(foreign_dependency,),
            resolved_run_spec=resolved,
            source_evidence=source,
            scientific_criteria=("scientific.reference",),
            performance_criteria=("performance.reference",),
        )
    assert provider.calls == []


def test_operational_evidence_cannot_satisfy_scientific_gate():
    route = "production-local"
    provider_id, deployment, dependency, resolved, context, source, matrix = _bindings(
        route
    )
    operational_impostor = QualificationEvidence(
        "operational",
        "passed",
        (resolved.spec_id, dependency.dependency_id),
        build_id=context.build_id,
        environment_id=context.environment_id,
        backend=context.backend,
        topology=context.topology,
        precision=context.precision,
        reduction=context.reduction,
        replay_id=context.replay_id,
        criteria_ids=("scientific.reference",),
        raw_artifact_ids=("raw.operational-impostor",),
        reviewer_id=context.reviewer_id,
        issued_at=10,
        expires_at=90,
        reason="Operational evidence cannot stand in for scientific evidence.",
    )
    artifact = produce_provider_qualification(
        _InjectedProvider(provider_id, deployment),
        matrix,
        context=context,
        support_dependencies=(dependency,),
        resolved_run_spec=resolved,
        source_evidence=(operational_impostor, source[1]),
        scientific_criteria=("scientific.reference",),
        performance_criteria=("performance.reference",),
    )

    assert artifact["gates"][0]["gate"] == "scientific"
    assert artifact["gates"][0]["outcome"] == "inconclusive"
    assert artifact["gates"][1]["outcome"] == "passed"
    assert artifact["status"] == "inconclusive"
    verify_provider_qualification(artifact)


def test_expired_scientific_evidence_is_not_current():
    route = "production-local"
    provider_id, deployment, dependency, resolved, context, source, matrix = _bindings(
        route
    )
    expired = QualificationEvidence(
        "scientific",
        "passed",
        (resolved.spec_id, dependency.dependency_id),
        build_id=context.build_id,
        environment_id=context.environment_id,
        backend=context.backend,
        topology=context.topology,
        precision=context.precision,
        reduction=context.reduction,
        replay_id=context.replay_id,
        criteria_ids=("scientific.reference",),
        raw_artifact_ids=("raw.expired-science",),
        reviewer_id=context.reviewer_id,
        issued_at=1,
        expires_at=19,
        reason="Expired evidence must not qualify a current candidate.",
    )
    artifact = produce_provider_qualification(
        _InjectedProvider(provider_id, deployment),
        matrix,
        context=context,
        support_dependencies=(dependency,),
        resolved_run_spec=resolved,
        source_evidence=(expired, source[1]),
        scientific_criteria=("scientific.reference",),
        performance_criteria=("performance.reference",),
    )
    assert artifact["gates"][0]["outcome"] == "inconclusive"
    reasons = artifact["gates"][0]["coverage"]["gaps"][0]["reasons"]
    assert any(reason.startswith("expired-evidence:") for reason in reasons)


def test_artifacts_are_deterministic_provider_specific_and_tamper_evident():
    first, _ = _produce("durable-service")
    second, _ = _produce("durable-service")
    assert first == second

    (
        provider_id,
        deployment,
        dependency,
        resolved,
        context,
        source,
        matrix,
    ) = _bindings("durable-service", deployment_id="deployment.other")
    other = produce_provider_qualification(
        _InjectedProvider(provider_id, deployment),
        matrix,
        context=context,
        support_dependencies=(dependency,),
        resolved_run_spec=resolved,
        source_evidence=source,
        scientific_criteria=("scientific.reference",),
        performance_criteria=("performance.reference",),
    )
    assert first["artifact_id"] != other["artifact_id"]

    tampered = deepcopy(first)
    tampered["provider_id"] = "provider.tampered"
    with pytest.raises(ValueError, match="content address"):
        verify_provider_qualification(tampered)
