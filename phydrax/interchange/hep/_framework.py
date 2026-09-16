#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...measurement._operations import OperationalCoordinate, ResolvedConditionSnapshot


def _identifiers(
    values: Sequence[str], name: str, /, *, required: bool = True
) -> tuple[str, ...]:
    result = tuple(str(value).strip() for value in values)
    if (
        (required and not result)
        or any(not value for value in result)
        or len(set(result)) != len(result)
    ):
        raise ValueError(f"{name} must contain distinct non-empty values.")
    return tuple(sorted(result))


class DataTierReference(StrictModule, NonTrainableState):
    experiment_id: str = eqx.field(static=True)
    tier_id: str = eqx.field(static=True)
    dataset_id: str = eqx.field(static=True)
    format_profile_id: str = eqx.field(static=True)
    resource_ids: tuple[str, ...] = eqx.field(static=True)
    conditions_snapshot_id: str = eqx.field(static=True)
    reference_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        experiment_id: str,
        tier_id: str,
        dataset_id: str,
        format_profile_id: str,
        resource_ids: Sequence[str],
        conditions_snapshot_id: str,
    ):
        values = tuple(
            str(value).strip()
            for value in (
                experiment_id,
                tier_id,
                dataset_id,
                format_profile_id,
                conditions_snapshot_id,
            )
        )
        resources = _identifiers(resource_ids, "resource_ids")
        if any(not value for value in values):
            raise ValueError("Data-tier identities must be non-empty.")
        (
            self.experiment_id,
            self.tier_id,
            self.dataset_id,
            self.format_profile_id,
            self.conditions_snapshot_id,
        ) = values
        self.resource_ids = resources
        self.reference_id = canonical_fingerprint(
            {
                "kind": "hep-data-tier-reference",
                "values": list(values),
                "resources": list(resources),
            }
        )


class FrameworkEventContext(StrictModule, NonTrainableState):
    coordinate: OperationalCoordinate
    conditions: ResolvedConditionSnapshot
    framework_id: str = eqx.field(static=True)
    framework_release: str = eqx.field(static=True)
    execution_slot: int = eqx.field(static=True)
    input_collection_ids: tuple[str, ...] = eqx.field(static=True)
    output_collection_ids: tuple[str, ...] = eqx.field(static=True)
    device_id: str = eqx.field(static=True)
    stream_id: str = eqx.field(static=True)
    side_effects: tuple[str, ...] = eqx.field(static=True)
    context_id: str = eqx.field(static=True)

    def __init__(
        self,
        coordinate: OperationalCoordinate,
        conditions: ResolvedConditionSnapshot,
        /,
        *,
        framework_id: str,
        framework_release: str,
        execution_slot: int,
        input_collection_ids: Sequence[str],
        output_collection_ids: Sequence[str],
        device_id: str,
        stream_id: str,
        side_effects: Sequence[str] = ("none",),
    ):
        if not isinstance(coordinate, OperationalCoordinate):
            raise TypeError("coordinate must be OperationalCoordinate.")
        if (
            not isinstance(conditions, ResolvedConditionSnapshot)
            or conditions.coordinate.coordinate_id != coordinate.coordinate_id
        ):
            raise ValueError("conditions must resolve the exact framework coordinate.")
        values = tuple(
            str(value).strip()
            for value in (framework_id, framework_release, device_id, stream_id)
        )
        if any(not value for value in values) or int(execution_slot) < 0:
            raise ValueError("Framework identity, device, stream, and slot are invalid.")
        inputs = _identifiers(input_collection_ids, "input_collection_ids")
        outputs = _identifiers(output_collection_ids, "output_collection_ids")
        effects = _identifiers(side_effects, "side_effects")
        allowed_effects = {"none", "framework-read", "framework-write"}
        if not set(effects) <= allowed_effects or (
            "none" in effects and len(effects) != 1
        ):
            raise ValueError("Framework side effects are invalid.")
        self.coordinate = coordinate
        self.conditions = conditions
        self.framework_id, self.framework_release, self.device_id, self.stream_id = values
        self.execution_slot = int(execution_slot)
        self.input_collection_ids = inputs
        self.output_collection_ids = outputs
        self.side_effects = effects
        self.context_id = canonical_fingerprint(
            {
                "kind": "hep-framework-event-context",
                "coordinate": coordinate.coordinate_id,
                "conditions": conditions.snapshot_id,
                "framework": values[:2],
                "slot": self.execution_slot,
                "inputs": list(inputs),
                "outputs": list(outputs),
                "device": self.device_id,
                "stream": self.stream_id,
                "side_effects": list(effects),
            }
        )


__all__ = ["DataTierReference", "FrameworkEventContext"]
