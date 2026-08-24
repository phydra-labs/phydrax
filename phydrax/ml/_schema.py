#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Literal, TypeAlias

import equinox as eqx

from .._strict import StrictModule
from .._trainable import NonTrainableState


FeatureKind: TypeAlias = Literal["continuous", "ordinal", "categorical", "boolean"]
TargetKind: TypeAlias = Literal[
    "continuous", "binary", "multiclass", "multilabel", "ordinal", "count", "ranking"
]


class FeatureSchema(StrictModule, NonTrainableState):
    """Static identity and semantic kind for a canonical feature axis."""

    names: tuple[str, ...] = eqx.field(static=True)
    kinds: tuple[FeatureKind, ...] = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        names: Sequence[str],
        /,
        *,
        kinds: Sequence[FeatureKind] | None = None,
        layout_id: str = "",
    ):
        names_ = tuple(str(name) for name in names)
        if not names_ or any(not name for name in names_):
            raise ValueError("Feature names must be non-empty strings.")
        if len(set(names_)) != len(names_):
            raise ValueError("Feature names must be unique.")
        kinds_ = ("continuous",) * len(names_) if kinds is None else tuple(kinds)
        valid_kinds = {"continuous", "ordinal", "categorical", "boolean"}
        if len(kinds_) != len(names_) or any(kind not in valid_kinds for kind in kinds_):
            raise ValueError(
                "Feature kinds must align with names and use supported values."
            )
        self.names = names_
        self.kinds = kinds_
        self.layout_id = str(layout_id)

    @classmethod
    def anonymous(cls, size: int, /) -> "FeatureSchema":
        count = int(size)
        if count <= 0:
            raise ValueError("Feature count must be positive.")
        return cls(tuple(f"feature_{index}" for index in range(count)))

    def select(self, indices: Sequence[int], /) -> "FeatureSchema":
        selected = tuple(int(index) for index in indices)
        if any(index < 0 or index >= len(self.names) for index in selected):
            raise IndexError("Feature selection contains an out-of-range index.")
        return FeatureSchema(
            tuple(self.names[index] for index in selected),
            kinds=tuple(self.kinds[index] for index in selected),
            layout_id=self.layout_id,
        )


class TargetSchema(StrictModule, NonTrainableState):
    """Static target semantics and optional external class vocabulary."""

    kind: TargetKind = eqx.field(static=True)
    names: tuple[str, ...] = eqx.field(static=True)
    class_labels: tuple[object, ...] = eqx.field(static=True)

    def __init__(
        self,
        kind: TargetKind = "continuous",
        /,
        *,
        names: Sequence[str] = (),
        class_labels: Sequence[object] = (),
    ):
        valid = {
            "continuous",
            "binary",
            "multiclass",
            "multilabel",
            "ordinal",
            "count",
            "ranking",
        }
        if kind not in valid:
            raise ValueError(f"Unsupported target kind {kind!r}.")
        names_ = tuple(str(name) for name in names)
        if any(not name for name in names_) or len(set(names_)) != len(names_):
            raise ValueError("Target names must be non-empty and unique.")
        labels_ = tuple(class_labels)
        if kind == "binary" and labels_ and len(labels_) != 2:
            raise ValueError("Binary target schemas require exactly two class labels.")
        if kind == "multiclass" and labels_ and len(labels_) < 2:
            raise ValueError("Multiclass target schemas require at least two labels.")
        if kind == "multilabel" and not names_:
            raise ValueError("Multilabel target schemas require named label coordinates.")
        if kind == "ordinal":
            if len(labels_) < 3:
                raise ValueError("Ordinal target schemas require at least three labels.")
            if not all(isinstance(label, Hashable) for label in labels_) or len(
                set(labels_)
            ) != len(labels_):
                raise ValueError("Ordinal class labels must be hashable and unique.")
        self.kind = kind
        self.names = names_
        self.class_labels = labels_

    @property
    def num_classes(self) -> int:
        return len(self.class_labels)

    @property
    def num_labels(self) -> int:
        return len(self.names) if self.kind == "multilabel" else 0


__all__ = ["FeatureKind", "FeatureSchema", "TargetKind", "TargetSchema"]
