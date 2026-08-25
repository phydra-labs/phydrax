#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._resources import CliffordResourceBudget


class CliffordAlgebraSpec(StrictModule, NonTrainableState):
    """Immutable diagonal real Clifford algebra and basis convention."""

    diagonal: tuple[int, ...] = eqx.field(static=True)
    basis_labels: tuple[str, ...] = eqx.field(static=True)
    orientation: int = eqx.field(static=True)
    blade_order: str = eqx.field(static=True)
    budget: CliffordResourceBudget
    algebra_id: str = eqx.field(static=True)
    spec_id: str = eqx.field(static=True)

    def __init__(
        self,
        diagonal: Sequence[int],
        /,
        *,
        basis_labels: Sequence[str] | None = None,
        orientation: int = 1,
        budget: CliffordResourceBudget | None = None,
    ):
        entries = tuple(diagonal)
        if not entries:
            raise ValueError("Clifford algebra dimension must be positive.")
        if any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in entries
        ):
            raise TypeError("Clifford signature entries must be integers in {-1, 0, 1}.")
        resolved = tuple(int(value) for value in entries)
        if any(value not in (-1, 0, 1) for value in resolved):
            raise ValueError("Clifford signature entries must lie in {-1, 0, 1}.")
        if basis_labels is None:
            labels = tuple(f"e{axis}" for axis in range(len(resolved)))
        else:
            labels = tuple(str(value) for value in basis_labels)
            if len(labels) != len(resolved):
                raise ValueError(
                    "Clifford basis labels must match the algebra dimension."
                )
            if any(not value for value in labels):
                raise ValueError("Clifford basis labels must be non-empty.")
            if len(set(labels)) != len(labels):
                raise ValueError("Clifford basis labels must be unique.")
        orientation_ = int(orientation)
        if orientation_ not in (-1, 1) or orientation_ != orientation:
            raise ValueError("Clifford orientation must be +1 or -1.")
        budget_ = CliffordResourceBudget() if budget is None else budget
        if not isinstance(budget_, CliffordResourceBudget):
            raise TypeError("budget must be a CliffordResourceBudget or None.")
        blade_order = "grade-lexicographic-v1"
        self.diagonal = resolved
        self.basis_labels = labels
        self.orientation = orientation_
        self.blade_order = blade_order
        self.budget = budget_
        self.algebra_id = canonical_fingerprint(
            {
                "kind": "clifford-algebra-v1",
                "diagonal": list(resolved),
                "basis_labels": list(labels),
                "blade_order": blade_order,
            }
        )
        self.spec_id = canonical_fingerprint(
            {
                "kind": "clifford-algebra-spec-v1",
                "algebra": self.algebra_id,
                "orientation": orientation_,
                "budget": budget_.budget_id,
            }
        )

    @classmethod
    def from_inertia(
        cls,
        positive: int,
        negative: int,
        radical: int = 0,
        /,
        **kwargs,
    ) -> "CliffordAlgebraSpec":
        raw_counts = (positive, negative, radical)
        if any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in raw_counts
        ):
            raise TypeError("Clifford inertia counts must be integers.")
        counts = tuple(int(value) for value in raw_counts)
        if any(value < 0 for value in counts) or sum(counts) <= 0:
            raise ValueError(
                "Clifford inertia counts must be nonnegative with positive total."
            )
        return cls(
            (1,) * counts[0] + (-1,) * counts[1] + (0,) * counts[2],
            **kwargs,
        )

    @property
    def dimension(self) -> int:
        return len(self.diagonal)

    @property
    def blade_count(self) -> int:
        return 1 << self.dimension

    @property
    def positive(self) -> int:
        return self.diagonal.count(1)

    @property
    def negative(self) -> int:
        return self.diagonal.count(-1)

    @property
    def radical(self) -> int:
        return self.diagonal.count(0)

    @property
    def nondegenerate(self) -> bool:
        return self.radical == 0

    @property
    def positive_definite(self) -> bool:
        return self.positive == self.dimension

    def to_dict(self) -> dict[str, object]:
        return {
            "diagonal": list(self.diagonal),
            "basis_labels": list(self.basis_labels),
            "orientation": self.orientation,
            "budget": {
                "maximum_blades": self.budget.maximum_blades,
                "maximum_product_terms": self.budget.maximum_product_terms,
                "maximum_plan_bytes": self.budget.maximum_plan_bytes,
                "maximum_dense_kernel_bytes": self.budget.maximum_dense_kernel_bytes,
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object], /) -> "CliffordAlgebraSpec":
        budget_value = value["budget"]
        if not isinstance(budget_value, Mapping):
            raise TypeError("Serialized Clifford budget must be a mapping.")
        budget = CliffordResourceBudget(
            maximum_blades=int(budget_value["maximum_blades"]),
            maximum_product_terms=int(budget_value["maximum_product_terms"]),
            maximum_plan_bytes=int(budget_value["maximum_plan_bytes"]),
            maximum_dense_kernel_bytes=int(budget_value["maximum_dense_kernel_bytes"]),
        )
        diagonal = value["diagonal"]
        labels = value["basis_labels"]
        if not isinstance(diagonal, Sequence) or isinstance(diagonal, (str, bytes)):
            raise TypeError("Serialized Clifford diagonal must be a sequence.")
        if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)):
            raise TypeError("Serialized Clifford labels must be a sequence.")
        return cls(
            diagonal,
            basis_labels=tuple(str(item) for item in labels),
            orientation=int(value["orientation"]),
            budget=budget,
        )

    def require_compatible(self, other: "CliffordAlgebraSpec", /) -> None:
        if not isinstance(other, CliffordAlgebraSpec):
            raise TypeError("Expected a CliffordAlgebraSpec.")
        if self.algebra_id != other.algebra_id:
            raise ValueError("Clifford algebra specifications do not match.")


__all__ = ["CliffordAlgebraSpec"]
