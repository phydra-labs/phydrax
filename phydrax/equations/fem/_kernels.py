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
    reference_ids: tuple[str, ...] = eqx.field(static=True)
    element_ids: tuple[str, ...] = eqx.field(static=True)
    coordinate_element_ids: tuple[str, ...] = eqx.field(static=True)
    representations: tuple[str, ...] = eqx.field(static=True)
    mappings: tuple[str, ...] = eqx.field(static=True)
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
        reference_ids: Sequence[str],
        element_ids: Sequence[str],
        coordinate_element_ids: Sequence[str],
        representations: Sequence[str],
        mappings: Sequence[str],
        coefficient_layout_ids: Sequence[str],
        precision_id: str,
        ir_semantics_id: str,
    ):
        identifier = str(kernel_id)
        kind = str(kernel_kind)
        strategy = str(local_kernel)
        references = tuple(sorted(set(str(value) for value in reference_ids)))
        elements = tuple(sorted(set(str(value) for value in element_ids)))
        coordinate_elements = tuple(
            sorted(set(str(value) for value in coordinate_element_ids))
        )
        representations_ = tuple(sorted(set(str(value) for value in representations)))
        mappings_ = tuple(sorted(set(str(value) for value in mappings)))
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
            *elements,
            *coordinate_elements,
            *representations_,
            *mappings_,
            *layouts,
        )
        if (
            any(not value for value in identities)
            or not references
            or not elements
            or not coordinate_elements
            or not representations_
            or not mappings_
            or (evaluator is not None and not callable(evaluator))
        ):
            raise ValueError("Kernel binding compilation identities are incomplete.")
        self.kernel_id = identifier
        self.kernel_kind = kind
        self.local_kernel = strategy
        self.reference_ids = references
        self.element_ids = elements
        self.coordinate_element_ids = coordinate_elements
        self.representations = representations_
        self.mappings = mappings_
        self.coefficient_layout_ids = layouts
        self.precision_id = precision
        self.ir_semantics_id = semantics
        self.evaluator = evaluator
        self.binding_id = canonical_fingerprint(
            {
                "kind": "finite-element-kernel-binding",
                "kernel_id": identifier,
                "kernel_kind": kind,
                "local_kernel": strategy,
                "references": references,
                "elements": elements,
                "coordinate_elements": coordinate_elements,
                "representations": representations_,
                "mappings": mappings_,
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
                "kind": "finite-element-kernel-table",
                "bindings": [value.binding_id for value in bindings_],
            }
        )

    def binding(self, kernel_id: str, /) -> KernelBinding:
        requested = str(kernel_id)
        for value in self.bindings:
            if value.kernel_id == requested:
                return value
        raise KeyError(f"Unknown finite-element kernel {requested!r}.")


__all__ = ["KernelBinding", "KernelTable"]
