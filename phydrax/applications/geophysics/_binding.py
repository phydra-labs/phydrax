# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Application bindings refer to native storage; they do not own numerical arrays."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ..._fingerprint import canonical_fingerprint, canonical_json
from ..._model import ValuePort
from ._quantities import GeophysicalQuantity
from ._time import TemporalSupport
from ._vertical import HybridPressureCoordinate


def _owner_dimensions_verified(
    quantity: GeophysicalQuantity,
    port: ValuePort,
    components: tuple[str, ...],
    storage_kind: str,
    /,
) -> bool:
    """Check the quantity dimension against every bound owner component.

    Returns `False` when the owner port declares no dimensions; a declared
    dimension that differs from `quantity.unit.dimension` raises `ValueError`.
    """
    if port.dimensions is None:
        return False
    declared = dict(zip(port.component_ids, port.dimensions, strict=True))
    mismatched = tuple(
        component
        for component in components or port.component_ids
        if declared[component] != quantity.unit.dimension
    )
    if mismatched:
        raise ValueError(
            f"Geophysical quantity {quantity.name!r} dimension does not match the "
            f"dimensions declared by {storage_kind} owner port {port.semantic_id!r} "
            f"for components {mismatched}."
        )
    return True


@dataclass(frozen=True, slots=True, init=False)
class GeophysicalFieldBinding:
    """Reference from one geophysical quantity to exactly one native storage owner.

    The quantity dimension is checked against the owner's `ValuePort` for the
    bound components (the selected state components, the complete field space,
    or the named operator task field): a declared owner dimension must equal
    `quantity.unit.dimension`, otherwise construction raises `ValueError`.
    `dimensions_verified` is `True` only when the owner declared every bound
    dimension; state layouts and discrete field spaces declare none, so their
    bindings record `False`. It is derived from the owner and is not part of the
    archived descriptor or `binding_id`.
    """

    quantity: GeophysicalQuantity
    storage_kind: str
    storage_id: str
    components: tuple[str, ...]
    field_name: str | None
    role: str
    vertical_id: str | None
    temporal: TemporalSupport | None
    binding_id: str = field(init=False)
    dimensions_verified: bool = field(init=False)
    _vertical_json: str | None = field(init=False, repr=False)

    def __init__(
        self,
        quantity: GeophysicalQuantity,
        *,
        field_space: Any = None,
        state_layout: Any = None,
        components: tuple[str, ...] = (),
        operator_task: Any = None,
        field_name: str | None = None,
        role: str = "prognostic",
        vertical: HybridPressureCoordinate | None = None,
        temporal: TemporalSupport | None = None,
    ):
        from ...discretization import DiscreteFieldSpace
        from ...dynamics import StateLayout
        from ...nn.operator.task import OperatorTask

        if not isinstance(quantity, GeophysicalQuantity):
            raise TypeError("Binding quantity must be a GeophysicalQuantity.")
        if (
            sum(owner is not None for owner in (field_space, state_layout, operator_task))
            != 1
        ):
            raise ValueError("A binding must refer to exactly one native storage owner.")
        if role not in (
            "prognostic",
            "diagnostic",
            "forcing",
            "observation",
            "flux",
            "parameter",
        ):
            raise ValueError("Unknown geophysical field role.")
        if isinstance(components, str):
            raise TypeError("Components must be a sequence of names, not one string.")
        components = tuple(components)
        if len(set(components)) != len(components):
            raise ValueError("Binding components must be unique.")
        if state_layout is not None:
            if not isinstance(state_layout, StateLayout):
                raise TypeError("state_layout must be a native StateLayout.")
            if not components or not set(components).issubset(
                state_layout.component_names
            ):
                raise ValueError(
                    "A state binding requires an exact nonempty native component selection."
                )
            if field_name is not None:
                raise ValueError(
                    "State bindings use components, not operator field names."
                )
            storage_kind, storage_id = "state-layout", state_layout.layout_id
            owner_port = state_layout.value_port()
        elif field_space is not None:
            if not isinstance(field_space, DiscreteFieldSpace):
                raise TypeError("field_space must be a native DiscreteFieldSpace.")
            if components or field_name is not None:
                raise ValueError(
                    "Field-space bindings refer to the complete field space."
                )
            storage_kind, storage_id = "field-space", field_space.field_space_id
            owner_port = field_space.value_port()
        else:
            if not isinstance(operator_task, OperatorTask):
                raise TypeError("operator_task must be a native OperatorTask.")
            if components or field_name not in operator_task.field_by_name:
                raise ValueError("Operator binding requires one exact task field name.")
            storage_kind, storage_id = "operator-task", operator_task.fingerprint
            owner_port = operator_task.field_by_name[field_name].value_port()
        dimensions_verified = _owner_dimensions_verified(
            quantity, owner_port, components, storage_kind
        )
        if vertical is not None and not isinstance(vertical, HybridPressureCoordinate):
            raise TypeError("vertical must be a prepared HybridPressureCoordinate.")
        if temporal is not None and not isinstance(temporal, TemporalSupport):
            raise TypeError("temporal must be TemporalSupport.")
        vertical_payload = None if vertical is None else vertical.to_dict()
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "storage_kind", storage_kind)
        object.__setattr__(self, "storage_id", storage_id)
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "field_name", field_name)
        object.__setattr__(self, "role", role)
        object.__setattr__(
            self, "vertical_id", None if vertical is None else vertical.coordinate_id
        )
        object.__setattr__(self, "temporal", temporal)
        object.__setattr__(self, "dimensions_verified", dimensions_verified)
        object.__setattr__(
            self,
            "_vertical_json",
            None if vertical is None else canonical_json(vertical_payload),
        )
        object.__setattr__(self, "binding_id", canonical_fingerprint(self._payload()))

    def _payload(self) -> dict[str, Any]:
        return {
            "quantity": self.quantity.to_dict(),
            "storage": {
                "kind": self.storage_kind,
                "id": self.storage_id,
                "components": list(self.components),
                "field_name": self.field_name,
            },
            "role": self.role,
            "vertical": None
            if self._vertical_json is None
            else json.loads(self._vertical_json),
            "temporal": None if self.temporal is None else self.temporal.to_dict(),
        }

    @property
    def descriptor(self) -> dict[str, Any]:
        return self.to_dict()

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "binding_id": self.binding_id}

    def value_port(self) -> ValuePort:
        """Return the port of the bound quantity's selected state components.

        `semantic_id` is `f"geophysical-quantity:{quantity.quantity_id}"`, so the
        quantity name, kind, unit and its scale, axes, sign convention, support,
        and reference configuration all enter the identity. The event is the
        selected component vector: `event_shape` is `(len(components),)` and
        `component_ids` are the selected state-layout component names in binding
        order. Every component carries the quantity's declared dimension
        `quantity.unit.dimension`. The representation is the fixed literal
        `"geophysical-field"`; space, frame, normalization, and event axes are
        undeclared (the quantity `axes` index samples, not event components);
        variance is neutral.

        Raises `ValueError` for field-space and operator-task bindings, which
        declare no component layout of their own; their ports come from the
        native storage owner.
        """
        if self.storage_kind != "state-layout":
            raise ValueError(
                f"A {self.storage_kind} geophysical binding declares no component "
                "layout; derive its port from the native storage owner."
            )
        return ValuePort(
            f"geophysical-quantity:{self.quantity.quantity_id}",
            event_shape=(len(self.components),),
            component_ids=self.components,
            representation="geophysical-field",
            dimensions=(self.quantity.unit.dimension,) * len(self.components),
        )

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        field_space: Any = None,
        state_layout: Any = None,
        operator_task: Any = None,
    ) -> GeophysicalFieldBinding:
        """Rebind an archived descriptor only against an explicitly supplied native owner."""
        if set(payload) != {
            "quantity",
            "storage",
            "role",
            "vertical",
            "temporal",
            "binding_id",
        }:
            raise ValueError("Binding descriptor must use exactly the canonical fields.")
        content = {key: value for key, value in payload.items() if key != "binding_id"}
        if canonical_fingerprint(content) != payload["binding_id"]:
            raise ValueError("Binding descriptor fingerprint does not match its content.")
        storage = payload["storage"]
        if set(storage) != {"kind", "id", "components", "field_name"}:
            raise ValueError("Storage descriptor must use exactly the canonical fields.")
        binding = cls(
            GeophysicalQuantity.from_dict(payload["quantity"]),
            field_space=field_space,
            state_layout=state_layout,
            operator_task=operator_task,
            components=tuple(storage["components"]),
            field_name=storage["field_name"],
            role=payload["role"],
            vertical=None
            if payload["vertical"] is None
            else HybridPressureCoordinate.from_dict(payload["vertical"]),
            temporal=None
            if payload["temporal"] is None
            else TemporalSupport.from_dict(payload["temporal"]),
        )
        if binding.binding_id != payload["binding_id"]:
            raise ValueError(
                "Supplied native storage does not match the archived binding."
            )
        return binding


__all__ = ["GeophysicalFieldBinding"]
