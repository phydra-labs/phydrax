#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import coordax as cx
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from phydrax.conditions._ir import (
    AbstractConditionOperator,
    OperatorCapabilities,
    OperatorLinearization,
)
from phydrax.domain import DomainFunction, PointBatch

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule


class ObservationActionEvidence(StrictModule):
    """Structural evidence for one exact finite observation action."""

    action_id: str = eqx.field(static=True)
    field: str = eqx.field(static=True)
    observation_count: int = eqx.field(static=True)
    components: tuple[int, ...] | None = eqx.field(static=True)
    exact_scope: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        action_id: str,
        field: str,
        observation_count: int,
        components: tuple[int, ...] | None,
        exact_scope: str = "finite_restriction",
    ):
        count = int(observation_count)
        if count <= 0:
            raise ValueError("Observation actions require at least one observation.")
        self.action_id = str(action_id)
        self.field = str(field)
        self.observation_count = count
        self.components = components
        self.exact_scope = str(exact_scope)


def _validated_components(
    components: Sequence[int] | None,
    /,
) -> tuple[int, ...] | None:
    if components is None:
        return None
    out = tuple(int(component) for component in components)
    if not out:
        raise ValueError("components must be non-empty when provided.")
    if any(component < 0 for component in out):
        raise ValueError("components must contain non-negative indices.")
    if len(set(out)) != len(out):
        raise ValueError("components must not contain duplicate indices.")
    return out


def _point_axis_and_count(batch: PointBatch, /) -> tuple[str | None, int]:
    axes = batch.structure.axis_names
    if axes is None:
        raise ValueError("Point observation layout must be canonicalized.")
    if not axes:
        return None, 1
    if len(axes) != 1:
        raise ValueError(
            "Point observations require one coupled sampling axis; "
            "coordinate-separable batches do not define a finite point list."
        )
    axis = axes[0]
    count: int | None = None
    for field in batch.points.values():
        if not isinstance(field, cx.Field) or axis not in field.dims:
            continue
        size = int(field.data.shape[field.dims.index(axis)])
        if count is not None and count != size:
            raise ValueError("Point observation fields disagree on sampling-axis size.")
        count = size
    if count is None:
        raise ValueError("Point observation batch does not carry its sampling axis.")
    return axis, count


class PointObservationAction(AbstractConditionOperator):
    """Certified linear restriction of one field to an immutable point batch.

    The action is exact only for the declared finite batch. Any off-batch extension
    belongs to a correction provider and is deliberately not part of this action.
    """

    field: str = eqx.field(static=True)
    batch: PointBatch
    components: tuple[int, ...] | None = eqx.field(static=True)
    capabilities: OperatorCapabilities = eqx.field(static=True)
    evidence: ObservationActionEvidence

    def __init__(
        self,
        field: str,
        batch: PointBatch,
        /,
        *,
        components: Sequence[int] | None = None,
    ):
        field_ = str(field)
        if not field_:
            raise ValueError("Point observation field name must be non-empty.")
        if not isinstance(batch, PointBatch):
            raise TypeError("PointObservationAction requires a PointBatch.")
        components_ = _validated_components(components)
        _, count = _point_axis_and_count(batch)
        action_id = canonical_fingerprint(
            {
                "kind": "point-observation-action-v1",
                "field": field_,
                "components": components_,
                "batch": array_tree_fingerprint(batch.points),
                "layout": repr(batch.structure),
            }
        )
        self.field = field_
        self.batch = batch
        self.components = components_
        self.capabilities = OperatorCapabilities(is_linear=True)
        self.evidence = ObservationActionEvidence(
            action_id=action_id,
            field=field_,
            observation_count=count,
            components=components_,
        )

    @property
    def action_id(self) -> str:
        return self.evidence.action_id

    @property
    def observation_count(self) -> int:
        return self.evidence.observation_count

    def _apply(self, values: Mapping[str, Any], /, *, key=None, **kwargs: Any) -> Array:
        if self.field not in values:
            raise KeyError(f"Missing observed field {self.field!r}.")
        value = values[self.field]
        if not isinstance(value, DomainFunction):
            raise TypeError("Point observations act on DomainFunction values.")
        evaluated = value(self.batch, key=key, **kwargs)
        if not isinstance(evaluated, cx.Field):
            raise TypeError("Point observation evaluation must return a coordax.Field.")
        data = jnp.asarray(evaluated.data)
        axis, count = _point_axis_and_count(self.batch)
        if axis is None:
            data = jnp.expand_dims(data, axis=0)
        else:
            if axis not in evaluated.dims:
                raise ValueError(
                    "Point observation output is missing the declared sampling axis."
                )
            data = jnp.moveaxis(data, evaluated.dims.index(axis), 0)
        if int(data.shape[0]) != count:
            raise ValueError(
                f"Point observation output has {data.shape[0]} rows; expected {count}."
            )
        if self.components is None:
            return data
        if data.ndim < 2:
            raise ValueError("Observation components require a trailing event axis.")
        width = int(data.shape[-1])
        if any(component >= width for component in self.components):
            raise ValueError(
                f"Observation component indices {self.components!r} exceed width {width}."
            )
        return jnp.take(data, jnp.asarray(self.components, dtype=jnp.int32), axis=-1)

    def apply(
        self,
        values: Mapping[str, Any],
        /,
        *,
        key: Any | None = None,
        **kwargs: Any,
    ) -> Array:
        return self._apply(values, key=key, **kwargs)

    def linear_action(
        self,
        values: Mapping[str, Any],
        /,
        *,
        key: Any | None = None,
        **kwargs: Any,
    ) -> Array:
        return self._apply(values, key=key, **kwargs)

    def adjoint_action(
        self,
        value: Any,
        /,
        *,
        key: Any | None = None,
        **kwargs: Any,
    ) -> Mapping[str, Any]:
        del value, key, kwargs
        raise TypeError(
            "PointObservationAction has no canonical function-space adjoint; "
            "a correction or representation provider must supply one."
        )

    def linearize(
        self,
        values: Mapping[str, Any],
        /,
        *,
        key: Any | None = None,
        **kwargs: Any,
    ) -> OperatorLinearization:
        del values, key, kwargs
        raise TypeError("Globally linear observation actions do not linearize.")


__all__ = [
    "ObservationActionEvidence",
    "PointObservationAction",
]
