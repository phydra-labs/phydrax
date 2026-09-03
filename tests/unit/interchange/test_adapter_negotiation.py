#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import pytest

from phydrax.interchange._report import (
    AdapterCapability,
    AdapterFormatProfile,
    AdapterLoss,
    AdapterNegotiationResult,
    AdapterReport,
    AdapterRequirement,
    AdapterStatus,
    AdapterWaiver,
    compose_adapter_reports,
    negotiate_adapter,
)


def _interpretation_loss(
    *affected_capability_ids: str,
) -> AdapterLoss:
    return AdapterLoss(
        "coordinates.handedness",
        "import",
        "transformed",
        "converted to the target coordinate convention",
        changes_interpretation=True,
        affected_capability_ids=affected_capability_ids,
    )


def _stage(
    name: str,
    source_profile: AdapterFormatProfile,
    target_profile: AdapterFormatProfile,
    source_id: str,
    target_id: str,
    *,
    coordinate_mapping: tuple[str, ...] = (),
    preserved_fields: tuple[str, ...] = (),
    assumptions: tuple[str, ...] = (),
    losses: tuple[AdapterLoss, ...] = (),
    requirements: tuple[AdapterRequirement, ...] = (),
    capabilities: tuple[AdapterCapability, ...] = (),
    waivers: tuple[AdapterWaiver, ...] = (),
) -> AdapterReport:
    return AdapterReport(
        AdapterStatus.DECLARED_LOSS if losses else AdapterStatus.LOSSLESS,
        source_profile.format,
        target_profile.format,
        source_id=source_id,
        target_id=target_id,
        coordinate_mapping=coordinate_mapping,
        preserved_fields=preserved_fields,
        assumptions=assumptions,
        losses=losses,
        stage=name,
        source_profile=source_profile,
        target_profile=target_profile,
        requirements=requirements,
        capabilities=capabilities,
        waivers=waivers,
    )


def test_satisfied_requirements_negotiate_with_stable_order_and_identity():
    required = AdapterRequirement("coordinates", rationale="needed by the target")
    optional = AdapterRequirement("labels", required=False)
    coordinates = AdapterCapability("coordinates", detail="preserved exactly")
    labels = AdapterCapability("labels")

    first = negotiate_adapter((optional, required), (labels, coordinates))
    second = AdapterNegotiationResult((required, optional), (coordinates, labels))

    assert first.valid
    assert first.status == AdapterStatus.LOSSLESS
    assert first.missing_required == ()
    assert first.missing_optional == ()
    first_semantics = tuple(item.semantic_id for item in first.requirements)
    second_semantics = tuple(item.semantic_id for item in second.requirements)
    assert first_semantics == second_semantics
    assert first.negotiation_id == second.negotiation_id


def test_missing_required_capability_fails_negotiation():
    required = AdapterRequirement("topology")

    result = negotiate_adapter((required,), ())

    assert not result.valid
    assert result.status == AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC
    assert result.missing_required == (required,)


def test_missing_optional_capability_remains_explicit_without_failing():
    optional = AdapterRequirement(
        "annotations", required=False, rationale="useful but not required"
    )

    result = negotiate_adapter((optional,), ())

    assert result.valid
    assert result.status == AdapterStatus.DECLARED_LOSS
    assert result.missing_optional == (optional,)
    assert result.missing_required == ()


def test_interpretation_change_requires_a_waiver_for_the_exact_loss():
    loss = _interpretation_loss()
    other_loss = AdapterLoss(
        "metadata.note",
        "import",
        "dropped",
        "not represented by the target",
        changes_interpretation=True,
    )

    unwaived = negotiate_adapter((), (), losses=(loss,))
    wrong_waiver = negotiate_adapter(
        (), (), losses=(loss,), waivers=(AdapterWaiver(other_loss, "accepted"),)
    )
    waiver = AdapterWaiver(loss, "the caller selected the target convention")
    waived = negotiate_adapter((), (), losses=(loss,), waivers=(waiver,))

    assert not unwaived.valid
    assert unwaived.unwaived_losses == (loss,)
    assert not wrong_waiver.valid
    assert wrong_waiver.unused_waivers != ()
    assert waived.valid
    assert waived.waived_losses == (loss,)
    assert waived.unwaived_losses == ()


def test_report_composition_rejects_broken_identity_continuity():
    source = AdapterFormatProfile("source", qualifiers={"encoding": "text"})
    parsed = AdapterFormatProfile("parsed", qualifiers={"shape": "tree"})
    normalized = AdapterFormatProfile("normalized", qualifiers={"shape": "tree"})
    parse = _stage("parse", source, parsed, "source-id", "parsed-id")
    normalize = _stage(
        "normalize", parsed, normalized, "different-parsed-id", "normalized-id"
    )

    with pytest.raises(ValueError, match="identity continuity"):
        compose_adapter_reports((parse, normalize))


def test_report_composition_requires_exact_format_profile_continuity():
    source = AdapterFormatProfile("source")
    parsed_text = AdapterFormatProfile("parsed", qualifiers={"encoding": "text"})
    parsed_binary = AdapterFormatProfile(
        "parsed", qualifiers={"encoding": "binary"}
    )
    normalized = AdapterFormatProfile("normalized")
    parse = _stage("parse", source, parsed_text, "source-id", "parsed-id")
    normalize = _stage(
        "normalize", parsed_binary, normalized, "parsed-id", "normalized-id"
    )

    with pytest.raises(ValueError, match="format-profile continuity"):
        compose_adapter_reports((parse, normalize))


def test_valid_report_chain_produces_one_deterministic_cumulative_report():
    source = AdapterFormatProfile("source", qualifiers={"encoding": "text"})
    parsed = AdapterFormatProfile("parsed", qualifiers={"shape": "tree"})
    normalized = AdapterFormatProfile("normalized", qualifiers={"units": "declared"})
    lowered = AdapterFormatProfile("lowered", qualifiers={"layout": "indexed"})
    backend = AdapterFormatProfile("backend", qualifiers={"storage": "native"})
    requirement = AdapterRequirement("coordinates")
    capability = AdapterCapability("coordinates")
    convention = AdapterCapability("coordinate-convention")
    loss = _interpretation_loss(convention.capability_id)
    waiver = AdapterWaiver(loss, "the caller selected the target convention")
    parse = _stage(
        "parse",
        source,
        parsed,
        "source-id",
        "parsed-id",
        coordinate_mapping=("source axes -> parsed axes",),
        preserved_fields=("values",),
        assumptions=("source is internally consistent",),
    )
    normalize = _stage(
        "normalize",
        parsed,
        normalized,
        "parsed-id",
        "normalized-id",
        preserved_fields=("values", "labels"),
        assumptions=("source is internally consistent",),
        requirements=(requirement,),
        capabilities=(capability,),
    )
    lower = _stage(
        "lower",
        normalized,
        lowered,
        "normalized-id",
        "lowered-id",
        coordinate_mapping=("normalized axes -> lowered axes",),
        losses=(loss,),
        waivers=(waiver,),
    )
    prepare_backend = _stage(
        "backend",
        lowered,
        backend,
        "lowered-id",
        "backend-id",
        preserved_fields=("labels",),
    )

    first = compose_adapter_reports((parse, normalize, lower, prepare_backend))
    second = compose_adapter_reports((parse, normalize, lower, prepare_backend))

    assert first.valid
    assert first.status == AdapterStatus.DECLARED_LOSS
    assert first.source_profile == source
    assert first.target_profile == backend
    assert first.coordinate_mapping == (
        "source axes -> parsed axes",
        "normalized axes -> lowered axes",
    )
    assert first.preserved_fields == ("values", "labels")
    assert first.assumptions == ("source is internally consistent",)
    assert first.losses == (loss,)
    assert first.negotiation.valid
    assert tuple(stage.report_id for stage in first.stages) == (
        parse.report_id,
        normalize.report_id,
        lower.report_id,
        prepare_backend.report_id,
    )
    assert first.report_id == second.report_id


def test_required_executable_capability_loss_cannot_be_waived():
    requirement = AdapterRequirement("execution")
    capability = AdapterCapability("execution", detail="native executable state")
    loss = AdapterLoss(
        "state.accepted",
        "import",
        "dropped",
        "accepted execution state would be discarded",
        changes_interpretation=True,
        affected_capability_ids=(capability.capability_id,),
    )
    waiver = AdapterWaiver(loss, "caller attempted to accept the loss")

    result = negotiate_adapter(
        (requirement,), (capability,), losses=(loss,), waivers=(waiver,)
    )

    assert not result.valid
    assert result.status == AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC
    assert result.waived_losses == ()
    assert result.unwaived_losses == (loss,)
    assert loss.affected_capability_ids == (capability.capability_id,)


def test_unassociated_loss_cannot_bypass_required_capability_protection():
    requirement = AdapterRequirement("execution")
    capability = AdapterCapability("execution")
    loss = _interpretation_loss()
    waiver = AdapterWaiver(loss, "accepted without identifying affected capability")

    result = negotiate_adapter(
        (requirement,), (capability,), losses=(loss,), waivers=(waiver,)
    )

    assert not result.valid
    assert result.waived_losses == ()
    assert result.unwaived_losses == (loss,)


def test_stale_and_semantically_unused_waivers_reject():
    stale_loss = _interpretation_loss()
    stale_waiver = AdapterWaiver(stale_loss, "no matching declared loss")
    informational_loss = AdapterLoss(
        "metadata.note",
        "import",
        "dropped",
        "informational metadata is omitted",
        changes_interpretation=False,
    )
    unnecessary_waiver = AdapterWaiver(
        informational_loss, "no interpretation change needed acceptance"
    )

    stale = negotiate_adapter((), (), waivers=(stale_waiver,))
    unnecessary = negotiate_adapter(
        (), (), losses=(informational_loss,), waivers=(unnecessary_waiver,)
    )

    assert not stale.valid
    assert stale.unused_waivers == (stale_waiver,)
    assert not unnecessary.valid
    assert unnecessary.unused_waivers == (unnecessary_waiver,)


def test_composition_derives_optional_and_required_negotiation_status():
    source = AdapterFormatProfile("source")
    target = AdapterFormatProfile("target")
    optional = AdapterRequirement("annotations", required=False)
    required = AdapterRequirement("executable-state")
    optional_stage = _stage(
        "parse",
        source,
        target,
        "source-id",
        "target-id",
        requirements=(optional,),
    )
    required_stage = _stage(
        "parse",
        source,
        target,
        "other-source-id",
        "other-target-id",
        requirements=(required,),
    )

    optional_report = compose_adapter_reports((optional_stage,))
    required_report = compose_adapter_reports((required_stage,))

    assert optional_report.valid
    assert optional_report.status == AdapterStatus.DECLARED_LOSS
    assert optional_report.losses == ()
    assert optional_report.negotiation.missing_optional == (optional,)
    assert not required_report.valid
    assert required_report.status == AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC
    assert required_report.negotiation.missing_required == (required,)
