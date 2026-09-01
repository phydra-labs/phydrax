#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class KernelBinding(StrictModule, NonTrainableState):
    kernel_id: str = eqx.field(static=True)
    kernel_kind: str = eqx.field(static=True)
    local_kernel: str = eqx.field(static=True)
    reference_action_ids: tuple[str, ...] = eqx.field(static=True)
    field_layout_ids: tuple[str, ...] = eqx.field(static=True)
    geometry_action_ids: tuple[str, ...] = eqx.field(static=True)
    coefficient_layout_ids: tuple[str, ...] = eqx.field(static=True)
    precision_id: str = eqx.field(static=True)
    ir_semantics_id: str = eqx.field(static=True)
    evaluator: Callable | None
    binding_id: str = eqx.field(static=True)

    def __init__(
        self,
        kernel_id: str,
        kernel_kind: str,
        evaluator: Callable | None = None,
        /,
        *,
        local_kernel: str,
        reference_action_ids: Sequence[str],
        field_layout_ids: Sequence[str],
        geometry_action_ids: Sequence[str],
        coefficient_layout_ids: Sequence[str],
        precision_id: str,
        ir_semantics_id: str,
    ):
        identifier = str(kernel_id)
        kind = str(kernel_kind)
        strategy = str(local_kernel)
        references = tuple(sorted(set(str(value) for value in reference_action_ids)))
        fields = tuple(sorted(set(str(value) for value in field_layout_ids)))
        geometries = tuple(sorted(set(str(value) for value in geometry_action_ids)))
        layouts = tuple(sorted(set(str(value) for value in coefficient_layout_ids)))
        precision = str(precision_id)
        semantics = str(ir_semantics_id)
        identities = (
            identifier,
            kind,
            strategy,
            precision,
            semantics,
            *references,
            *fields,
            *geometries,
            *layouts,
        )
        if (
            any(not value for value in identities)
            or not references
            or not fields
            or not geometries
            or (evaluator is not None and not callable(evaluator))
        ):
            raise ValueError("Kernel binding compilation identities are incomplete.")
        self.kernel_id = identifier
        self.kernel_kind = kind
        self.local_kernel = strategy
        self.reference_action_ids = references
        self.field_layout_ids = fields
        self.geometry_action_ids = geometries
        self.coefficient_layout_ids = layouts
        self.precision_id = precision
        self.ir_semantics_id = semantics
        self.evaluator = evaluator
        self.binding_id = canonical_fingerprint(
            {
                "kind": "local-kernel-binding",
                "kernel_id": identifier,
                "kernel_kind": kind,
                "local_kernel": strategy,
                "reference_actions": references,
                "field_layouts": fields,
                "geometry_actions": geometries,
                "coefficient_layouts": layouts,
                "precision": precision,
                "ir_semantics": semantics,
                "has_evaluator": evaluator is not None,
            }
        )


class KernelTable(StrictModule, NonTrainableState):
    bindings: tuple[KernelBinding, ...]
    table_id: str = eqx.field(static=True)

    def __init__(self, bindings: Sequence[KernelBinding], /):
        bindings_ = tuple(bindings)
        if not bindings_ or not all(
            isinstance(value, KernelBinding) for value in bindings_
        ):
            raise ValueError("KernelTable requires one or more KernelBinding values.")
        identifiers = tuple(value.kernel_id for value in bindings_)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("KernelTable kernel IDs must be unique.")
        self.bindings = bindings_
        self.table_id = canonical_fingerprint(
            {
                "kind": "local-kernel-table",
                "bindings": [value.binding_id for value in bindings_],
            }
        )

    def binding(self, kernel_id: str, /) -> KernelBinding:
        requested = str(kernel_id)
        for value in self.bindings:
            if value.kernel_id == requested:
                return value
        raise KeyError(f"Unknown local kernel {requested!r}.")


__all__ = ["KernelBinding", "KernelTable"]
