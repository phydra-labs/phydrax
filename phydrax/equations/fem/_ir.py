#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


FieldSlotRole = Literal[
    "unknown",
    "cell-local",
    "trace",
    "coefficient",
    "quadrature-state",
    "test",
    "trial",
    "control",
]
RegionKind = Literal[
    "cell",
    "exterior-facet",
    "interior-facet",
    "interface",
    "smoothing-patch",
    "embedded-volume",
    "embedded-interface",
    "contact-pair",
]
DifferentialOperator = Literal[
    "value",
    "grad",
    "sym-grad",
    "div",
    "curl",
    "normal-trace",
    "tangential-trace",
    "jump",
    "average",
    "smoothed-grad",
    "shape-average",
    "primitive-moment",
]
ActionKind = Literal[
    "residual",
    "energy",
    "bilinear",
    "linear",
    "functional",
    "material",
]


class FieldSlot(StrictModule, NonTrainableState):
    name: str = eqx.field(static=True)
    role: FieldSlotRole = eqx.field(static=True)
    space_id: str = eqx.field(static=True)
    value_shape: tuple[int, ...] = eqx.field(static=True)
    slot_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        role: FieldSlotRole,
        space_id: str,
        /,
        *,
        value_shape: Sequence[int] = (),
    ):
        name_ = str(name)
        space = str(space_id)
        shape = tuple(int(size) for size in value_shape)
        if not name_ or not space or any(size <= 0 for size in shape):
            raise ValueError("Field slot name, space, or value shape is invalid.")
        if role not in (
            "unknown",
            "cell-local",
            "trace",
            "coefficient",
            "quadrature-state",
            "test",
            "trial",
            "control",
        ):
            raise ValueError("Unknown field slot role.")
        self.name = name_
        self.role = role
        self.space_id = space
        self.value_shape = shape
        self.slot_id = canonical_fingerprint(
            {
                "kind": "finite-element-field-slot",
                "name": name_,
                "role": role,
                "space": space,
                "value_shape": list(shape),
            }
        )


class RegionIR(StrictModule, NonTrainableState):
    region_kind: RegionKind = eqx.field(static=True)
    domain_id: str = eqx.field(static=True)
    rule_ids: tuple[tuple[str, str], ...] = eqx.field(static=True)
    region_id: str = eqx.field(static=True)

    def __init__(
        self,
        region_kind: RegionKind,
        domain_id: str,
        rule_ids: Sequence[tuple[str, str]],
        /,
    ):
        domain = str(domain_id)
        rules = tuple(sorted((str(block), str(rule)) for block, rule in rule_ids))
        if not domain or any(not block or not rule for block, rule in rules):
            raise ValueError("Region domain/rule identities must be non-empty.")
        if region_kind not in (
            "cell",
            "exterior-facet",
            "interior-facet",
            "interface",
            "smoothing-patch",
            "embedded-volume",
            "embedded-interface",
            "contact-pair",
        ):
            raise ValueError("Unknown integration region kind.")
        self.region_kind = region_kind
        self.domain_id = domain
        self.rule_ids = rules
        self.region_id = canonical_fingerprint(
            {
                "kind": "finite-element-region-ir",
                "region_kind": region_kind,
                "domain": domain,
                "rules": [list(item) for item in rules],
            }
        )


class FiniteElementActionIR(StrictModule, NonTrainableState):
    action_kind: ActionKind = eqx.field(static=True)
    output_slot: str = eqx.field(static=True)
    input_slots: tuple[str, ...] = eqx.field(static=True)
    operators: tuple[tuple[str, DifferentialOperator], ...] = eqx.field(static=True)
    region: RegionIR
    kernel_id: str = eqx.field(static=True)
    action_id: str = eqx.field(static=True)

    def __init__(
        self,
        action_kind: ActionKind,
        output_slot: str,
        input_slots: Sequence[str],
        operators: Sequence[tuple[str, DifferentialOperator]],
        region: RegionIR,
        kernel_id: str,
        /,
    ):
        output = str(output_slot)
        inputs = tuple(str(name) for name in input_slots)
        operations = tuple((str(name), operation) for name, operation in operators)
        kernel = str(kernel_id)
        if not output or not inputs or any(not name for name in inputs) or not kernel:
            raise ValueError("Local action field/kernel identifiers must be non-empty.")
        if len(set(inputs)) != len(inputs):
            raise ValueError("Local action input slots must be unique.")
        if not isinstance(region, RegionIR):
            raise TypeError("region must be RegionIR.")
        self.action_kind = action_kind
        self.output_slot = output
        self.input_slots = inputs
        self.operators = operations
        self.region = region
        self.kernel_id = kernel
        self.action_id = canonical_fingerprint(
            {
                "kind": "finite-element-action",
                "action_kind": action_kind,
                "output": output,
                "inputs": list(inputs),
                "operators": [list(item) for item in operations],
                "region": region.region_id,
                "kernel": kernel,
            }
        )


class LocalActionIR(StrictModule, NonTrainableState):
    slots: tuple[FieldSlot, ...]
    actions: tuple[FiniteElementActionIR, ...]
    ir_id: str = eqx.field(static=True)

    def __init__(
        self,
        slots: Sequence[FieldSlot],
        actions: Sequence[FiniteElementActionIR],
        /,
    ):
        slots_ = tuple(slots)
        actions_ = tuple(actions)
        if not slots_ or not actions_:
            raise ValueError("LocalActionIR requires slots and actions.")
        names = tuple(slot.name for slot in slots_)
        if len(set(names)) != len(names):
            raise ValueError("LocalActionIR slot names must be unique.")
        declared = set(names)
        for action in actions_:
            if action.output_slot not in declared or any(
                name not in declared for name in action.input_slots
            ):
                raise ValueError("Finite-element action references an undeclared slot.")
        self.slots = slots_
        self.actions = actions_
        self.ir_id = canonical_fingerprint(
            {
                "kind": "finite-element-local-action-ir",
                "slots": [slot.slot_id for slot in slots_],
                "actions": [action.action_id for action in actions_],
            }
        )


__all__ = [
    "ActionKind",
    "DifferentialOperator",
    "FieldSlot",
    "FieldSlotRole",
    "LocalActionIR",
    "FiniteElementActionIR",
    "RegionIR",
    "RegionKind",
]
