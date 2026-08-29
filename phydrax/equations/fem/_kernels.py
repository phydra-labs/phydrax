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
    evaluator: Callable | None
    binding_id: str = eqx.field(static=True)

    def __init__(
        self,
        kernel_id: str,
        kernel_kind: str,
        evaluator: Callable | None = None,
        /,
    ):
        identifier = str(kernel_id)
        kind = str(kernel_kind)
        if (
            not identifier
            or not kind
            or (evaluator is not None and not callable(evaluator))
        ):
            raise ValueError("Kernel binding identity, kind, or evaluator is invalid.")
        self.kernel_id = identifier
        self.kernel_kind = kind
        self.evaluator = evaluator
        self.binding_id = canonical_fingerprint(
            {
                "kind": "finite-element-kernel-binding",
                "kernel_id": identifier,
                "kernel_kind": kind,
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
