#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import IntEnum
from typing import final, Literal

import equinox as eqx

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


_AdapterDirection = Literal["import", "export"]
_AdapterLossCategory = Literal[
    "dropped", "synthesized", "transformed", "unsupported"
]


class AdapterStatus(IntEnum):
    """Outcome of an external-format conversion."""

    LOSSLESS = 0
    DECLARED_LOSS = 1
    UNSUPPORTED_REQUIRED_SEMANTIC = 2
    MALFORMED_SOURCE = 3
    OPTIONAL_DEPENDENCY_UNAVAILABLE = 4
    INCONSISTENT_SOURCE = 5


@final
class AdapterLoss(StrictModule, NonTrainableState):
    """One exact semantic change made by an external-format adapter."""

    path: str = eqx.field(static=True)
    direction: _AdapterDirection = eqx.field(static=True)
    category: _AdapterLossCategory = eqx.field(static=True)
    rationale: str = eqx.field(static=True)
    changes_interpretation: bool = eqx.field(static=True)
    affected_capability_ids: tuple[str, ...] = eqx.field(static=True)
    loss_id: str = eqx.field(static=True)

    def __init__(
        self,
        path: str,
        direction: _AdapterDirection,
        category: _AdapterLossCategory,
        rationale: str,
        /,
        *,
        changes_interpretation: bool,
        affected_capability_ids: Sequence[str] = (),
    ):
        path_ = str(path).strip()
        rationale_ = str(rationale).strip()
        if not path_ or not rationale_:
            raise ValueError("Adapter loss paths and rationales must be non-empty.")
        if direction not in ("import", "export"):
            raise ValueError("Adapter loss direction must be 'import' or 'export'.")
        if category not in ("dropped", "synthesized", "transformed", "unsupported"):
            raise ValueError("Unknown adapter loss category.")
        affected_capability_ids_ = tuple(
            sorted(_strings(affected_capability_ids, "affected_capability_ids"))
        )
        self.path = path_
        self.direction = direction
        self.category = category
        self.rationale = rationale_
        self.changes_interpretation = bool(changes_interpretation)
        self.affected_capability_ids = affected_capability_ids_
        self.loss_id = canonical_fingerprint(
            {
                "kind": "adapter-loss",
                "path": path_,
                "direction": direction,
                "category": category,
                "rationale": rationale_,
                "changes_interpretation": self.changes_interpretation,
                "affected_capability_ids": list(affected_capability_ids_),
            }
        )


@final
class AdapterFormatProfile(StrictModule, NonTrainableState):
    """Exact immutable description of one adapter endpoint format."""

    format: str = eqx.field(static=True)
    qualifiers: tuple[tuple[str, str], ...] = eqx.field(static=True)
    profile_id: str = eqx.field(static=True)

    def __init__(
        self,
        format: str,
        /,
        *,
        qualifiers: Mapping[str, str] | Sequence[tuple[str, str]] = (),
    ):
        format_ = str(format).strip()
        if not format_:
            raise ValueError("Adapter format profiles require a non-empty format.")
        qualifiers_ = _string_pairs(qualifiers, "qualifiers")
        self.format = format_
        self.qualifiers = qualifiers_
        self.profile_id = canonical_fingerprint(
            {
                "kind": "adapter-format-profile",
                "format": format_,
                "qualifiers": [list(item) for item in qualifiers_],
            }
        )


@final
class AdapterRequirement(StrictModule, NonTrainableState):
    """One semantic requested from an adapter target."""

    semantic_id: str = eqx.field(static=True)
    required: bool = eqx.field(static=True)
    rationale: str = eqx.field(static=True)
    requirement_id: str = eqx.field(static=True)

    def __init__(
        self,
        semantic_id: str,
        /,
        *,
        required: bool = True,
        rationale: str = "",
    ):
        semantic_id_ = str(semantic_id).strip()
        rationale_ = str(rationale).strip()
        if not semantic_id_:
            raise ValueError("Adapter requirement semantic IDs must be non-empty.")
        self.semantic_id = semantic_id_
        self.required = bool(required)
        self.rationale = rationale_
        self.requirement_id = canonical_fingerprint(
            {
                "kind": "adapter-requirement",
                "semantic_id": semantic_id_,
                "required": self.required,
                "rationale": rationale_,
            }
        )


@final
class AdapterCapability(StrictModule, NonTrainableState):
    """One semantic that an adapter target can preserve."""

    semantic_id: str = eqx.field(static=True)
    detail: str = eqx.field(static=True)
    capability_id: str = eqx.field(static=True)

    def __init__(self, semantic_id: str, /, *, detail: str = ""):
        semantic_id_ = str(semantic_id).strip()
        detail_ = str(detail).strip()
        if not semantic_id_:
            raise ValueError("Adapter capability semantic IDs must be non-empty.")
        self.semantic_id = semantic_id_
        self.detail = detail_
        self.capability_id = canonical_fingerprint(
            {
                "kind": "adapter-capability",
                "semantic_id": semantic_id_,
                "detail": detail_,
            }
        )


@final
class AdapterWaiver(StrictModule, NonTrainableState):
    """Explicit acceptance of one interpretation-changing adapter loss."""

    loss_id: str = eqx.field(static=True)
    rationale: str = eqx.field(static=True)
    waiver_id: str = eqx.field(static=True)

    def __init__(self, loss: AdapterLoss | str, rationale: str, /):
        loss_id_ = loss.loss_id if isinstance(loss, AdapterLoss) else str(loss).strip()
        rationale_ = str(rationale).strip()
        if not loss_id_ or not rationale_:
            raise ValueError("Adapter waivers require a loss ID and rationale.")
        self.loss_id = loss_id_
        self.rationale = rationale_
        self.waiver_id = canonical_fingerprint(
            {
                "kind": "adapter-waiver",
                "loss_id": loss_id_,
                "rationale": rationale_,
            }
        )


@final
class AdapterNegotiationResult(StrictModule, NonTrainableState):
    """Deterministic result of matching requested and available semantics."""

    valid: bool = eqx.field(static=True)
    status: AdapterStatus = eqx.field(static=True)
    requirements: tuple[AdapterRequirement, ...] = eqx.field(static=True)
    capabilities: tuple[AdapterCapability, ...] = eqx.field(static=True)
    losses: tuple[AdapterLoss, ...] = eqx.field(static=True)
    waivers: tuple[AdapterWaiver, ...] = eqx.field(static=True)
    satisfied_requirements: tuple[AdapterRequirement, ...] = eqx.field(static=True)
    missing_required: tuple[AdapterRequirement, ...] = eqx.field(static=True)
    missing_optional: tuple[AdapterRequirement, ...] = eqx.field(static=True)
    waived_losses: tuple[AdapterLoss, ...] = eqx.field(static=True)
    unwaived_losses: tuple[AdapterLoss, ...] = eqx.field(static=True)
    unused_waivers: tuple[AdapterWaiver, ...] = eqx.field(static=True)
    negotiation_id: str = eqx.field(static=True)

    def __init__(
        self,
        requirements: Sequence[AdapterRequirement],
        capabilities: Sequence[AdapterCapability],
        /,
        *,
        losses: Sequence[AdapterLoss] = (),
        waivers: Sequence[AdapterWaiver] = (),
    ):
        requirements_ = _requirements(requirements)
        capabilities_ = _capabilities(capabilities)
        losses_ = _losses(losses)
        waivers_ = _waivers(waivers)
        available = frozenset(item.semantic_id for item in capabilities_)
        waived_loss_ids = frozenset(item.loss_id for item in waivers_)
        required_semantics = frozenset(
            item.semantic_id for item in requirements_ if item.required
        )
        required_capability_ids = frozenset(
            item.capability_id
            for item in capabilities_
            if item.semantic_id in required_semantics
        )
        unwaivable_loss_ids = frozenset(
            item.loss_id
            for item in losses_
            if item.changes_interpretation
            and (
                required_capability_ids.intersection(
                    item.affected_capability_ids
                )
                or (
                    required_capability_ids
                    and not item.affected_capability_ids
                )
            )
        )
        satisfied = tuple(
            item for item in requirements_ if item.semantic_id in available
        )
        missing_required = tuple(
            item
            for item in requirements_
            if item.required and item.semantic_id not in available
        )
        missing_optional = tuple(
            item
            for item in requirements_
            if not item.required and item.semantic_id not in available
        )
        waived_losses = tuple(
            item
            for item in losses_
            if item.changes_interpretation
            and item.loss_id in waived_loss_ids
            and item.loss_id not in unwaivable_loss_ids
        )
        unwaived_losses = tuple(
            item
            for item in losses_
            if item.changes_interpretation
            and (
                item.loss_id not in waived_loss_ids
                or item.loss_id in unwaivable_loss_ids
            )
        )
        losses_by_id = {item.loss_id: item for item in losses_}
        unused_waivers = tuple(
            item
            for item in waivers_
            if item.loss_id not in losses_by_id
            or not losses_by_id[item.loss_id].changes_interpretation
        )
        if missing_required or unwaived_losses or unused_waivers:
            status = AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC
        elif losses_ or missing_optional:
            status = AdapterStatus.DECLARED_LOSS
        else:
            status = AdapterStatus.LOSSLESS
        self.valid = status in (AdapterStatus.LOSSLESS, AdapterStatus.DECLARED_LOSS)
        self.status = status
        self.requirements = requirements_
        self.capabilities = capabilities_
        self.losses = losses_
        self.waivers = waivers_
        self.satisfied_requirements = satisfied
        self.missing_required = missing_required
        self.missing_optional = missing_optional
        self.waived_losses = waived_losses
        self.unwaived_losses = unwaived_losses
        self.unused_waivers = unused_waivers
        self.negotiation_id = canonical_fingerprint(
            {
                "kind": "adapter-negotiation-result",
                "requirements": [item.requirement_id for item in requirements_],
                "capabilities": [item.capability_id for item in capabilities_],
                "losses": [item.loss_id for item in losses_],
                "waivers": [item.waiver_id for item in waivers_],
                "status": int(status),
            }
        )


@final
class AdapterReport(StrictModule, NonTrainableState):
    """Auditable status and semantic accounting for one conversion."""

    valid: bool = eqx.field(static=True)
    status: AdapterStatus = eqx.field(static=True)
    stage: str = eqx.field(static=True)
    source_format: str = eqx.field(static=True)
    target_format: str = eqx.field(static=True)
    source_profile: AdapterFormatProfile = eqx.field(static=True)
    target_profile: AdapterFormatProfile = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    target_id: str = eqx.field(static=True)
    coordinate_mapping: tuple[str, ...] = eqx.field(static=True)
    preserved_fields: tuple[str, ...] = eqx.field(static=True)
    assumptions: tuple[str, ...] = eqx.field(static=True)
    losses: tuple[AdapterLoss, ...] = eqx.field(static=True)
    requirements: tuple[AdapterRequirement, ...] = eqx.field(static=True)
    capabilities: tuple[AdapterCapability, ...] = eqx.field(static=True)
    waivers: tuple[AdapterWaiver, ...] = eqx.field(static=True)
    negotiation: AdapterNegotiationResult = eqx.field(static=True)
    stages: tuple[AdapterReport, ...] = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    def __init__(
        self,
        status: AdapterStatus,
        source_format: str,
        target_format: str,
        /,
        *,
        source_id: str,
        target_id: str,
        coordinate_mapping: Sequence[str] = (),
        preserved_fields: Sequence[str] = (),
        assumptions: Sequence[str] = (),
        losses: Sequence[AdapterLoss] = (),
        stage: str = "adapter",
        source_profile: AdapterFormatProfile | None = None,
        target_profile: AdapterFormatProfile | None = None,
        requirements: Sequence[AdapterRequirement] = (),
        capabilities: Sequence[AdapterCapability] = (),
        waivers: Sequence[AdapterWaiver] = (),
        stages: Sequence[AdapterReport] = (),
    ):
        status_ = AdapterStatus(status)
        source_format_ = str(source_format).strip()
        target_format_ = str(target_format).strip()
        source_id_ = str(source_id).strip()
        target_id_ = str(target_id).strip()
        stage_ = str(stage).strip()
        if (
            not source_format_
            or not target_format_
            or not source_id_
            or not target_id_
            or not stage_
        ):
            raise ValueError(
                "Adapter formats, identities, and stage names must be non-empty."
            )
        losses_ = tuple(losses)
        if not all(isinstance(item, AdapterLoss) for item in losses_):
            raise TypeError("losses must contain AdapterLoss values.")
        if status_ == AdapterStatus.LOSSLESS and losses_:
            raise ValueError("A lossless report cannot contain semantic losses.")
        source_profile_ = (
            AdapterFormatProfile(source_format_)
            if source_profile is None
            else source_profile
        )
        target_profile_ = (
            AdapterFormatProfile(target_format_)
            if target_profile is None
            else target_profile
        )
        if not isinstance(source_profile_, AdapterFormatProfile) or not isinstance(
            target_profile_, AdapterFormatProfile
        ):
            raise TypeError(
                "source_profile and target_profile must be AdapterFormatProfile values."
            )
        if (
            source_profile_.format != source_format_
            or target_profile_.format != target_format_
        ):
            raise ValueError(
                "Adapter profile formats must match their report endpoint formats."
            )
        requirements_ = _requirements(requirements)
        capabilities_ = _capabilities(capabilities)
        waivers_ = _waivers(waivers)
        stages_ = tuple(stages)
        if not all(isinstance(item, AdapterReport) for item in stages_):
            raise TypeError("stages must contain AdapterReport values.")
        negotiation = AdapterNegotiationResult(
            requirements_,
            capabilities_,
            losses=_deduplicate_losses(losses_),
            waivers=waivers_,
        )
        if (
            status_ == AdapterStatus.DECLARED_LOSS
            and not losses_
            and not negotiation.missing_optional
        ):
            raise ValueError(
                "A declared-loss report must enumerate losses or missing optional "
                "requirements."
            )
        enforce_negotiation = (
            source_profile is not None
            or target_profile is not None
            or stage_ != "adapter"
            or bool(requirements_)
            or bool(capabilities_)
            or bool(waivers_)
            or bool(stages_)
        )
        valid_status = status_ in (AdapterStatus.LOSSLESS, AdapterStatus.DECLARED_LOSS)
        self.valid = (
            valid_status
            and (negotiation.valid or not enforce_negotiation)
            and all(item.valid for item in stages_)
        )
        self.status = status_
        self.stage = stage_
        self.source_format = source_format_
        self.target_format = target_format_
        self.source_profile = source_profile_
        self.target_profile = target_profile_
        self.source_id = source_id_
        self.target_id = target_id_
        self.coordinate_mapping = _strings(coordinate_mapping, "coordinate_mapping")
        self.preserved_fields = _strings(preserved_fields, "preserved_fields")
        self.assumptions = _strings(assumptions, "assumptions")
        self.losses = losses_
        self.requirements = requirements_
        self.capabilities = capabilities_
        self.waivers = waivers_
        self.negotiation = negotiation
        self.stages = stages_
        self.report_id = canonical_fingerprint(
            {
                "kind": "adapter-report",
                "status": int(status_),
                "stage": stage_,
                "source_profile": source_profile_.profile_id,
                "target_profile": target_profile_.profile_id,
                "source_id": source_id_,
                "target_id": target_id_,
                "coordinate_mapping": list(self.coordinate_mapping),
                "preserved_fields": list(self.preserved_fields),
                "assumptions": list(self.assumptions),
                "losses": [item.loss_id for item in losses_],
                "negotiation": negotiation.negotiation_id,
                "stages": [item.report_id for item in stages_],
            }
        )


class AdapterError(ValueError):
    """Conversion failure with a machine-readable adapter status."""

    status: AdapterStatus

    def __init__(self, status: AdapterStatus, message: str, /):
        self.status = AdapterStatus(status)
        super().__init__(str(message))


def negotiate_adapter(
    requirements: Sequence[AdapterRequirement],
    capabilities: Sequence[AdapterCapability],
    /,
    *,
    losses: Sequence[AdapterLoss] = (),
    waivers: Sequence[AdapterWaiver] = (),
) -> AdapterNegotiationResult:
    """Negotiate generic semantic requirements against advertised capabilities."""
    return AdapterNegotiationResult(
        requirements, capabilities, losses=losses, waivers=waivers
    )


def compose_adapter_reports(
    reports: Sequence[AdapterReport], /
) -> AdapterReport:
    """Compose a continuous adapter-stage chain into one cumulative report."""
    reports_ = tuple(reports)
    if not reports_:
        raise ValueError("Adapter report composition requires at least one report.")
    if not all(isinstance(item, AdapterReport) for item in reports_):
        raise TypeError("reports must contain AdapterReport values.")
    stages = _flatten_stages(reports_)
    stage_ids = tuple(item.report_id for item in stages)
    if len(set(stage_ids)) != len(stage_ids):
        raise ValueError("Adapter stage identities must be unique within a chain.")
    previous_rank = -1
    stage_ranks = {"parse": 0, "normalize": 1, "lower": 2, "backend": 3}
    for stage in stages:
        if stage.stage in stage_ranks:
            rank = stage_ranks[stage.stage]
            if rank < previous_rank:
                raise ValueError(
                    "Adapter stages must follow parse, normalize, lower, backend order."
                )
            previous_rank = rank
    for source, target in zip(stages, stages[1:]):
        if source.target_id != target.source_id:
            raise ValueError(
                "Adapter report chain has broken source-to-target identity continuity."
            )
        if not _same_profile(source.target_profile, target.source_profile):
            raise ValueError(
                "Adapter report chain has broken format-profile continuity."
            )
    requirements = _merge_requirements(
        tuple(item for stage in stages for item in stage.requirements)
    )
    capabilities = _merge_capabilities(
        tuple(item for stage in stages for item in stage.capabilities)
    )
    losses = _deduplicate_losses(
        tuple(item for stage in stages for item in stage.losses)
    )
    waivers = _merge_waivers(tuple(item for stage in stages for item in stage.waivers))
    negotiation = negotiate_adapter(
        requirements, capabilities, losses=losses, waivers=waivers
    )
    failed_stage = next((stage for stage in stages if not stage.valid), None)
    if failed_stage is not None:
        status = (
            failed_stage.status
            if failed_stage.status
            not in (AdapterStatus.LOSSLESS, AdapterStatus.DECLARED_LOSS)
            else AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC
        )
    elif negotiation.status == AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC:
        status = AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC
    elif negotiation.status == AdapterStatus.DECLARED_LOSS:
        status = AdapterStatus.DECLARED_LOSS
    else:
        status = AdapterStatus.LOSSLESS
    first = stages[0]
    last = stages[-1]
    return AdapterReport(
        status,
        first.source_format,
        last.target_format,
        source_id=first.source_id,
        target_id=last.target_id,
        coordinate_mapping=_deduplicate_strings(
            tuple(value for stage in stages for value in stage.coordinate_mapping)
        ),
        preserved_fields=_deduplicate_strings(
            tuple(value for stage in stages for value in stage.preserved_fields)
        ),
        assumptions=_deduplicate_strings(
            tuple(value for stage in stages for value in stage.assumptions)
        ),
        losses=losses,
        stage="composed",
        source_profile=first.source_profile,
        target_profile=last.target_profile,
        requirements=requirements,
        capabilities=capabilities,
        waivers=waivers,
        stages=stages,
    )


def require_lossless(report: AdapterReport, /) -> None:
    """Reject a conversion result whose report declares semantic loss."""
    if report.status != AdapterStatus.LOSSLESS or not report.valid:
        raise AdapterError(
            AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
            "The requested lossless conversion cannot represent every native semantic.",
        )


def _strings(values: Sequence[str], owner: str, /) -> tuple[str, ...]:
    result = tuple(str(value).strip() for value in values)
    if any(not value for value in result) or len(set(result)) != len(result):
        raise ValueError(f"{owner} must contain unique non-empty strings.")
    return result


def _string_pairs(
    values: Mapping[str, str] | Sequence[tuple[str, str]],
    owner: str,
    /,
) -> tuple[tuple[str, str], ...]:
    raw = tuple(values.items()) if isinstance(values, Mapping) else tuple(values)
    if any(
        not isinstance(item, Sequence)
        or isinstance(item, str)
        or len(item) != 2
        for item in raw
    ):
        raise TypeError(f"{owner} must contain string key-value pairs.")
    result = tuple(
        (str(item[0]).strip(), str(item[1]).strip()) for item in raw
    )
    keys = tuple(key for key, _ in result)
    if (
        any(not key or not value for key, value in result)
        or len(set(keys)) != len(keys)
    ):
        raise ValueError(f"{owner} must contain unique non-empty string pairs.")
    return tuple(sorted(result))


def _requirements(
    values: Sequence[AdapterRequirement], /
) -> tuple[AdapterRequirement, ...]:
    result = tuple(values)
    if not all(isinstance(item, AdapterRequirement) for item in result):
        raise TypeError("requirements must contain AdapterRequirement values.")
    semantics = tuple(item.semantic_id for item in result)
    if len(set(semantics)) != len(semantics):
        raise ValueError("requirements must contain unique semantic IDs.")
    return tuple(sorted(result, key=lambda item: item.requirement_id))


def _capabilities(
    values: Sequence[AdapterCapability], /
) -> tuple[AdapterCapability, ...]:
    result = tuple(values)
    if not all(isinstance(item, AdapterCapability) for item in result):
        raise TypeError("capabilities must contain AdapterCapability values.")
    semantics = tuple(item.semantic_id for item in result)
    if len(set(semantics)) != len(semantics):
        raise ValueError("capabilities must contain unique semantic IDs.")
    return tuple(sorted(result, key=lambda item: item.capability_id))


def _losses(values: Sequence[AdapterLoss], /) -> tuple[AdapterLoss, ...]:
    result = tuple(values)
    if not all(isinstance(item, AdapterLoss) for item in result):
        raise TypeError("losses must contain AdapterLoss values.")
    loss_ids = tuple(item.loss_id for item in result)
    if len(set(loss_ids)) != len(loss_ids):
        raise ValueError("losses must contain unique loss IDs.")
    return tuple(sorted(result, key=lambda item: item.loss_id))


def _waivers(values: Sequence[AdapterWaiver], /) -> tuple[AdapterWaiver, ...]:
    result = tuple(values)
    if not all(isinstance(item, AdapterWaiver) for item in result):
        raise TypeError("waivers must contain AdapterWaiver values.")
    loss_ids = tuple(item.loss_id for item in result)
    if len(set(loss_ids)) != len(loss_ids):
        raise ValueError("waivers must address unique loss IDs.")
    return tuple(sorted(result, key=lambda item: item.waiver_id))


def _deduplicate_strings(values: Sequence[str], /) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _deduplicate_losses(values: Sequence[AdapterLoss], /) -> tuple[AdapterLoss, ...]:
    return tuple({item.loss_id: item for item in values}.values())


def _merge_requirements(
    values: Sequence[AdapterRequirement], /
) -> tuple[AdapterRequirement, ...]:
    merged: dict[str, AdapterRequirement] = {}
    for value in values:
        previous = merged.get(value.semantic_id)
        if previous is not None and previous.requirement_id != value.requirement_id:
            raise ValueError(
                "Adapter stages declare ambiguous requirements for one semantic ID."
            )
        merged[value.semantic_id] = value
    return _requirements(tuple(merged.values()))


def _merge_capabilities(
    values: Sequence[AdapterCapability], /
) -> tuple[AdapterCapability, ...]:
    merged: dict[str, AdapterCapability] = {}
    for value in values:
        previous = merged.get(value.semantic_id)
        if previous is not None and previous.capability_id != value.capability_id:
            raise ValueError(
                "Adapter stages declare ambiguous capabilities for one semantic ID."
            )
        merged[value.semantic_id] = value
    return _capabilities(tuple(merged.values()))


def _merge_waivers(
    values: Sequence[AdapterWaiver], /
) -> tuple[AdapterWaiver, ...]:
    merged: dict[str, AdapterWaiver] = {}
    for value in values:
        previous = merged.get(value.loss_id)
        if previous is not None and previous.waiver_id != value.waiver_id:
            raise ValueError(
                "Adapter stages declare ambiguous waivers for one loss ID."
            )
        merged[value.loss_id] = value
    return _waivers(tuple(merged.values()))


def _same_profile(
    left: AdapterFormatProfile, right: AdapterFormatProfile, /
) -> bool:
    return (
        left.profile_id == right.profile_id
        and left.format == right.format
        and left.qualifiers == right.qualifiers
    )


def _flatten_stages(
    reports: Sequence[AdapterReport], /
) -> tuple[AdapterReport, ...]:
    flattened: list[AdapterReport] = []
    pending = list(reversed(reports))
    while pending:
        report = pending.pop()
        if report.stages:
            pending.extend(reversed(report.stages))
        else:
            flattened.append(report)
    return tuple(flattened)


__all__ = [
    "AdapterCapability",
    "AdapterError",
    "AdapterFormatProfile",
    "AdapterLoss",
    "AdapterNegotiationResult",
    "AdapterReport",
    "AdapterStatus",
    "AdapterRequirement",
    "require_lossless",
    "AdapterWaiver",
    "compose_adapter_reports",
    "negotiate_adapter",
]
