#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx

from phydrax._fingerprint import canonical_fingerprint
from phydrax._strict import StrictModule

from ._identifiers import _token


def _evidence_group(values: Sequence[str], name: str, /) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of qualification evidence IDs.")
    normalized = tuple(_token(value, name) for value in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must contain unique evidence IDs.")
    return tuple(sorted(normalized))


class FinanceEvidenceBinding(StrictModule):
    """Four disjoint qualification-evidence bindings for a finance result.

    This record names evidence; it deliberately does not infer, replace, or report a
    qualification decision.
    """

    data_evidence_ids: tuple[str, ...] = eqx.field(static=True)
    model_evidence_ids: tuple[str, ...] = eqx.field(static=True)
    numerical_evidence_ids: tuple[str, ...] = eqx.field(static=True)
    use_evidence_ids: tuple[str, ...] = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)

    def __init__(
        self,
        data_evidence_ids: Sequence[str],
        model_evidence_ids: Sequence[str],
        numerical_evidence_ids: Sequence[str],
        use_evidence_ids: Sequence[str],
        /,
    ):
        data = _evidence_group(data_evidence_ids, "data_evidence_ids")
        model = _evidence_group(model_evidence_ids, "model_evidence_ids")
        numerical = _evidence_group(numerical_evidence_ids, "numerical_evidence_ids")
        use = _evidence_group(use_evidence_ids, "use_evidence_ids")
        groups = (data, model, numerical, use)
        all_ids = tuple(identifier for group in groups for identifier in group)
        if not all_ids:
            raise ValueError(
                "finance evidence binding must bind at least one evidence ID."
            )
        if len(set(all_ids)) != len(all_ids):
            raise ValueError(
                "data, model, numerical, and use evidence-ID groups must be disjoint."
            )
        self.data_evidence_ids = data
        self.model_evidence_ids = model
        self.numerical_evidence_ids = numerical
        self.use_evidence_ids = use
        self.binding_id = canonical_fingerprint(
            {
                "kind": "finance_evidence_binding",
                "data_evidence_ids": list(data),
                "model_evidence_ids": list(model),
                "numerical_evidence_ids": list(numerical),
                "use_evidence_ids": list(use),
            }
        )


__all__ = ["FinanceEvidenceBinding"]
