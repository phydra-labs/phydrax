#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import dataclasses
import math
from collections.abc import Hashable, Iterable, Sequence
from typing import Any, Literal, Self, TypeAlias

import equinox as eqx

from .._model import AbstractArrayModel, ModelPorts, ValuePort
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..units._dimension import DimensionSignature


FeatureKind: TypeAlias = Literal["continuous", "ordinal", "categorical", "boolean"]
TargetKind: TypeAlias = Literal[
    "continuous", "binary", "multiclass", "multilabel", "ordinal", "count", "ranking"
]


def _dimensions(
    values: Iterable[DimensionSignature] | None, count: int, owner: str, /
) -> tuple[DimensionSignature, ...] | None:
    if values is None:
        return None
    dimensions = tuple(values)
    if any(not isinstance(value, DimensionSignature) for value in dimensions):
        raise TypeError(f"{owner} dimensions must be DimensionSignature values.")
    if len(dimensions) != count:
        raise ValueError(f"{owner} dimensions must give one signature per component.")
    return dimensions


class FeatureSchema(StrictModule, NonTrainableState):
    """Static identity, semantic kind, and optional dimensions of a feature axis.

    `dimensions`, when declared, give one physical `DimensionSignature` per
    feature; `None` leaves feature dimensions undeclared.
    """

    names: tuple[str, ...] = eqx.field(static=True)
    kinds: tuple[FeatureKind, ...] = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)
    dimensions: tuple[DimensionSignature, ...] | None = eqx.field(static=True)

    def __init__(
        self,
        names: Sequence[str],
        /,
        *,
        kinds: Sequence[FeatureKind] | None = None,
        layout_id: str = "",
        dimensions: Iterable[DimensionSignature] | None = None,
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
        self.dimensions = _dimensions(dimensions, len(names_), "Feature")

    @classmethod
    def anonymous(cls, size: int, /) -> "FeatureSchema":
        count = int(size)
        if count <= 0:
            raise ValueError("Feature count must be positive.")
        return cls(tuple(f"feature_{index}" for index in range(count)))

    def select(self, indices: Sequence[int], /) -> "FeatureSchema":
        selected = tuple(indices)
        if any(index < 0 or index >= len(self.names) for index in selected):
            raise IndexError("Feature selection contains an out-of-range index.")
        return FeatureSchema(
            tuple(self.names[index] for index in selected),
            kinds=tuple(self.kinds[index] for index in selected),
            layout_id=self.layout_id,
            dimensions=(
                None
                if self.dimensions is None
                else tuple(self.dimensions[index] for index in selected)
            ),
        )


class TargetSchema(StrictModule, NonTrainableState):
    """Static target semantics, optional class vocabulary, and target dimensions.

    `dimensions`, when declared, give one physical `DimensionSignature` per named
    target component, so they require `names`.
    """

    kind: TargetKind = eqx.field(static=True)
    names: tuple[str, ...] = eqx.field(static=True)
    class_labels: tuple[object, ...] = eqx.field(static=True)
    dimensions: tuple[DimensionSignature, ...] | None = eqx.field(static=True)

    def __init__(
        self,
        kind: TargetKind = "continuous",
        /,
        *,
        names: Sequence[str] = (),
        class_labels: Sequence[object] = (),
        dimensions: Iterable[DimensionSignature] | None = None,
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
        classification_kinds = {"binary", "multiclass", "ordinal"}
        if labels_ and kind not in classification_kinds:
            raise ValueError(f"{kind} target schemas cannot declare class_labels.")
        if labels_ and (
            not all(isinstance(label, Hashable) for label in labels_)
            or len(set(labels_)) != len(labels_)
        ):
            raise ValueError("Class labels must be hashable and unique.")
        if kind == "binary" and labels_ and len(labels_) != 2:
            raise ValueError("Binary target schemas require exactly two class labels.")
        if kind == "multiclass" and labels_ and len(labels_) < 2:
            raise ValueError("Multiclass target schemas require at least two labels.")
        if kind == "multilabel" and not names_:
            raise ValueError("Multilabel target schemas require named label coordinates.")
        if kind == "ordinal" and len(labels_) < 3:
            raise ValueError("Ordinal target schemas require at least three labels.")
        if dimensions is not None and not names_:
            raise ValueError("Target dimensions require named target components.")
        self.kind = kind
        self.names = names_
        self.class_labels = labels_
        self.dimensions = _dimensions(dimensions, len(names_), "Target")

    @property
    def num_classes(self) -> int:
        return len(self.class_labels)

    @property
    def num_labels(self) -> int:
        return len(self.names) if self.kind == "multilabel" else 0


def _target_components(
    schema: TargetSchema, size: int, /
) -> tuple[tuple[str, ...], tuple[DimensionSignature, ...] | None]:
    if schema.names:
        if len(schema.names) != size:
            raise ValueError(
                f"Target schema names {schema.names!r} do not describe an event of "
                f"size {size}."
            )
        return schema.names, schema.dimensions
    if schema.class_labels and len(schema.class_labels) == size:
        return tuple(str(label) for label in schema.class_labels), None
    if size == 1:
        return ("target",), None
    return tuple(f"target_{index}" for index in range(size)), None


def schema_port(
    schema: FeatureSchema | TargetSchema,
    /,
    *,
    event_shape: Sequence[int] | None = None,
) -> ValuePort:
    """Return the derived `ValuePort` view of one ML schema.

    A `FeatureSchema` describes one feature vector: its semantic ID is the
    layout ID (`"ml-features"` when empty), its components are the feature names,
    its representation records the feature kinds, and its dimensions are the
    declared feature dimensions. A `TargetSchema` needs the `event_shape` of the
    target-valued output it describes: named targets are its components (with
    their dimensions), otherwise a class vocabulary matching the event size, a
    single `"target"` component, or positional `target_<i>` components.
    """
    if isinstance(schema, FeatureSchema):
        if event_shape is not None:
            raise ValueError("Feature schema ports derive their own event shape.")
        return ValuePort(
            schema.layout_id or "ml-features",
            event_shape=(len(schema.names),),
            component_ids=schema.names,
            representation="ml-features:" + ",".join(schema.kinds),
            dimensions=schema.dimensions,
        )
    if isinstance(schema, TargetSchema):
        if event_shape is None:
            raise ValueError("Target schema ports require the output event shape.")
        shape = tuple(event_shape)
        components, dimensions = _target_components(schema, math.prod(shape))
        return ValuePort(
            "ml-target",
            event_shape=shape,
            component_ids=components,
            representation=f"ml-target:{schema.kind}",
            dimensions=dimensions,
        )
    raise TypeError("schema must be a FeatureSchema or TargetSchema.")


def _size_shape(size: int | tuple[int, ...] | str, /) -> tuple[int, ...]:
    if size == "scalar":
        return ()
    if isinstance(size, int):
        return (size,)
    return tuple(size)


class AbstractFittedModel(AbstractArrayModel):
    """Fitted native ML executable carrying the schemas it was fitted against.

    `feature_schema` describes the final input axis; `target_schema` records the
    supervision target of a supervised fit. `phydrax.ml.fit` binds both from the
    fitted batch. `model_ports` is the derived port view: the feature port as its
    input, and the output ports of families whose calls estimate the target.
    """

    feature_schema: FeatureSchema | None = eqx.field(default=None, kw_only=True)
    target_schema: TargetSchema | None = eqx.field(default=None, kw_only=True)

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        # Every instance holds its (initially unbound) schema fields, so fitted
        # executables flatten, and restore from artifacts, with one structure.
        self = super().__new__(cls, *args, **kwargs)
        object.__setattr__(self, "feature_schema", None)
        object.__setattr__(self, "target_schema", None)
        return self

    def bind_schemas(
        self,
        feature_schema: FeatureSchema,
        /,
        target_schema: TargetSchema | None = None,
    ) -> Self:
        """Return this executable with its unbound schemas filled in.

        Schemas the fit already bound (for example a learned class vocabulary)
        are kept; the feature schema must match an integer `in_size`.
        """
        if not isinstance(feature_schema, FeatureSchema):
            raise TypeError("feature_schema must be a FeatureSchema.")
        if target_schema is not None and not isinstance(target_schema, TargetSchema):
            raise TypeError("target_schema must be a TargetSchema or None.")
        if isinstance(self.in_size, int) and len(feature_schema.names) != self.in_size:
            raise ValueError(
                f"Feature schema has {len(feature_schema.names)} features; the fitted "
                f"executable consumes {self.in_size}."
            )
        changes = {}
        if self.feature_schema is None:
            changes["feature_schema"] = feature_schema
        if self.target_schema is None and target_schema is not None:
            changes["target_schema"] = target_schema
        if not changes:
            return self
        bound = object.__new__(type(self))
        for field in dataclasses.fields(self):
            object.__setattr__(
                bound,
                field.name,
                changes.get(field.name, object.__getattribute__(self, field.name)),
            )
        object.__setattr__(bound, "_strict_initialized", True)
        return bound

    def output_ports(self) -> tuple[ValuePort, ...]:
        """Derived output ports; empty unless the family's call estimates the target."""
        return ()

    def target_output_ports(self) -> tuple[ValuePort, ...]:
        """Target port view of this executable's output, when a target is bound."""
        if self.target_schema is None:
            return ()
        return (schema_port(self.target_schema, event_shape=_size_shape(self.out_size)),)

    def model_ports(self) -> ModelPorts:
        """Return the derived input and output ports of this fitted executable."""
        if self.feature_schema is None:
            raise ValueError(
                f"{type(self).__name__} has no bound feature schema; fit it with "
                "phydrax.ml.fit or call bind_schemas before requesting ports."
            )
        return ModelPorts(
            inputs=(schema_port(self.feature_schema),), outputs=self.output_ports()
        )


__all__ = [
    "AbstractFittedModel",
    "FeatureKind",
    "FeatureSchema",
    "TargetKind",
    "TargetSchema",
    "schema_port",
]
