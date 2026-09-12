#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Canonical node/edge/face gather-scatter execution over patch bucket views."""

from __future__ import annotations

from collections.abc import Sequence
from math import prod

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import ArraySpace, transpose
from ...sparse import EdgeRelation, SparseCoordinateOperator
from ._entities import VariablePatchEntityBucketView, VariablePatchEntityComplex


class VariablePatchEntityFieldState(StrictModule):
    """Canonical degree field values, independent of patch bucket/device placement."""

    complex: VariablePatchEntityComplex
    degree: int = eqx.field(static=True)
    component_shape: tuple[int, ...] = eqx.field(static=True)
    values: Array
    state_id: str = eqx.field(static=True)

    def __init__(
        self,
        complex: VariablePatchEntityComplex,
        degree: int,
        values: ArrayLike,
        /,
        *,
        component_shape: Sequence[int] = (),
    ):
        degree_ = int(degree)
        components = tuple(int(value) for value in component_shape)
        if (
            not isinstance(complex, VariablePatchEntityComplex)
            or degree_ < 0
            or degree_ > complex.complex.dimension
            or any(value <= 0 for value in components)
        ):
            raise ValueError("Variable patch entity field specification is invalid.")
        array = jnp.asarray(values)
        expected = (complex.capacity[degree_],) + components
        if array.shape != expected:
            raise ValueError("Variable patch entity values do not match degree capacity.")
        active = complex.complex.entities(degree_).active_mask.reshape(
            (complex.capacity[degree_],) + (1,) * len(components)
        )
        self.complex = complex
        self.degree = degree_
        self.component_shape = components
        self.values = jnp.where(active, array, jnp.zeros((), dtype=array.dtype))
        self.state_id = canonical_fingerprint(
            {
                "kind": "variable-patch-entity-field-state",
                "complex": complex.complex_id,
                "degree": degree_,
                "component_shape": components,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            }
        )


class VariablePatchEntityRoute(StrictModule, NonTrainableState):
    """One signed global-to-local entity route with exact sparse transpose."""

    view: VariablePatchEntityBucketView
    gather: SparseCoordinateOperator
    route_id: str = eqx.field(static=True)

    def __init__(
        self,
        complex: VariablePatchEntityComplex,
        view: VariablePatchEntityBucketView,
        dtype,
        /,
    ):
        if not isinstance(complex, VariablePatchEntityComplex) or not isinstance(
            view, VariablePatchEntityBucketView
        ):
            raise TypeError("Entity route requires a complex and bucket view.")
        if view.level != complex.level:
            raise ValueError("Entity route view belongs to another AMR level.")
        source_size = complex.capacity[view.degree]
        local_size = prod(view.global_indices.shape)
        valid = view.valid.reshape((-1,))
        relation = EdgeRelation(
            jnp.maximum(view.global_indices.reshape((-1,)), 0),
            jnp.arange(local_size, dtype=jnp.int32),
            source_size=source_size,
            target_size=local_size,
            valid=valid,
        )
        gather = SparseCoordinateOperator(
            relation,
            view.orientation_signs.reshape((-1,)),
            source=ArraySpace((source_size,), dtype=dtype),
            target=ArraySpace((local_size,), dtype=dtype),
            operator_id=canonical_fingerprint(
                {
                    "kind": "variable-patch-entity-gather",
                    "complex": complex.complex_id,
                    "view": view.view_id,
                    "dtype": str(jnp.dtype(dtype)),
                }
            ),
        )
        self.view = view
        self.gather = gather
        self.route_id = canonical_fingerprint(
            {
                "kind": "variable-patch-entity-route",
                "complex": complex.complex_id,
                "view": view.view_id,
                "gather": gather.operator_id,
            }
        )


class VariablePatchEntityExecutionPlan(StrictModule, NonTrainableState):
    """Signed gather/scatter execution for one cochain degree across all bucket views."""

    complex: VariablePatchEntityComplex
    degree: int = eqx.field(static=True)
    component_shape: tuple[int, ...] = eqx.field(static=True)
    dtype: object = eqx.field(static=True)
    routes: tuple[VariablePatchEntityRoute, ...]
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        complex: VariablePatchEntityComplex,
        degree: int,
        /,
        *,
        component_shape: Sequence[int] = (),
        dtype=jnp.float64,
    ):
        degree_ = int(degree)
        components = tuple(int(value) for value in component_shape)
        dtype_ = jnp.dtype(dtype)
        if (
            not isinstance(complex, VariablePatchEntityComplex)
            or degree_ < 0
            or degree_ > complex.complex.dimension
            or any(value <= 0 for value in components)
            or not jnp.issubdtype(dtype_, jnp.inexact)
        ):
            raise ValueError("Variable patch entity execution plan is invalid.")
        routes = tuple(
            VariablePatchEntityRoute(complex, view, dtype_)
            for view in complex.views
            if view.degree == degree_
        )
        if not routes:
            raise ValueError("Entity execution requires at least one bucket view.")
        self.complex = complex
        self.degree = degree_
        self.component_shape = components
        self.dtype = dtype_
        self.routes = routes
        self.plan_id = canonical_fingerprint(
            {
                "kind": "variable-patch-entity-execution-plan",
                "complex": complex.complex_id,
                "degree": degree_,
                "component_shape": components,
                "dtype": str(dtype_),
                "routes": [route.route_id for route in routes],
            }
        )

    @property
    def component_count(self) -> int:
        return prod(self.component_shape) if self.component_shape else 1

    def gather(
        self,
        state: VariablePatchEntityFieldState,
        /,
    ) -> tuple[Array, ...]:
        if (
            not isinstance(state, VariablePatchEntityFieldState)
            or state.complex.complex_id != self.complex.complex_id
            or state.degree != self.degree
            or state.component_shape != self.component_shape
            or state.values.dtype != self.dtype
        ):
            raise ValueError("Entity field state does not match execution plan.")
        flattened = state.values.reshape((-1, self.component_count))
        result = []
        for route in self.routes:
            values = jnp.stack(
                tuple(
                    route.gather.mv(flattened[:, component])
                    for component in range(self.component_count)
                ),
                axis=-1,
            ).reshape(route.view.global_indices.shape + self.component_shape)
            result.append(values)
        return tuple(result)

    def scatter(self, local_values: Sequence[ArrayLike], /) -> Array:
        values = tuple(jnp.asarray(value) for value in local_values)
        if len(values) != len(self.routes):
            raise ValueError("Entity scatter requires one patch tensor per route.")
        global_values = jnp.zeros(
            (self.complex.capacity[self.degree], self.component_count),
            dtype=self.dtype,
        )
        for route, value in zip(self.routes, values, strict=True):
            expected = route.view.global_indices.shape + self.component_shape
            if value.shape != expected:
                raise ValueError("Entity scatter tensor does not match its bucket view.")
            flattened = value.reshape((-1, self.component_count))
            scatter = transpose(route.gather)
            global_values = global_values + jnp.stack(
                tuple(
                    scatter.mv(flattened[:, component])
                    for component in range(self.component_count)
                ),
                axis=-1,
            )
        active = self.complex.complex.entities(self.degree).active_mask.reshape(
            (self.complex.capacity[self.degree], 1)
        )
        return jnp.where(active, global_values, 0.0).reshape(
            (self.complex.capacity[self.degree],) + self.component_shape
        )


__all__ = [
    "VariablePatchEntityExecutionPlan",
    "VariablePatchEntityFieldState",
    "VariablePatchEntityRoute",
]
