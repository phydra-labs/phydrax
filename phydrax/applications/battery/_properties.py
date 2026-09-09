#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._interpolation import apply_gather_stencil, rectilinear_stencil
from ..._strict import StrictModule
from ...operators.interpolation import InterpolationResult, linear_interpolate


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical identifier.")
    return value


def _support_bounds(values: ArrayLike, /) -> Array:
    bounds = jnp.asarray(values)
    if bounds.shape != (2,) or jnp.issubdtype(bounds.dtype, jnp.complexfloating):
        raise ValueError("Property support bounds must be two real scalars.")
    host = np.asarray(bounds, dtype=float)
    if not np.all(np.isfinite(host)) or not host[1] > host[0]:
        raise ValueError(
            "Property support bounds must be finite and strictly increasing."
        )
    return bounds.astype(jnp.result_type(bounds, float))


def _value_bounds(values: ArrayLike, dtype, /) -> Array:
    bounds = jnp.asarray(values, dtype=dtype)
    if bounds.shape != (2,) or jnp.issubdtype(bounds.dtype, jnp.complexfloating):
        raise ValueError("Property value bounds must be two real scalars.")
    host = np.asarray(bounds, dtype=float)
    if np.any(np.isnan(host)) or not host[1] >= host[0]:
        raise ValueError(
            "Property value bounds must be ordered and may only use infinite endpoints."
        )
    return bounds


class ConstantPropertyLaw(StrictModule):
    """Dynamic scalar value restricted to one declared coordinate support interval."""

    value: Array
    support_bounds: Array
    value_bounds: Array
    quantity: str = eqx.field(static=True)
    coordinate: str = eqx.field(static=True)
    value_unit: str = eqx.field(static=True)
    coordinate_unit: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)

    def __init__(
        self,
        value: ArrayLike,
        support_bounds: ArrayLike,
        /,
        *,
        value_bounds: ArrayLike = (-jnp.inf, jnp.inf),
        quantity: str,
        coordinate: str,
        value_unit: str,
        coordinate_unit: str,
        source_id: str,
    ):
        value_ = jnp.asarray(value)
        if value_.shape != () or jnp.issubdtype(value_.dtype, jnp.complexfloating):
            raise ValueError("Constant property value must be one real scalar.")
        if not jnp.issubdtype(value_.dtype, jnp.inexact):
            value_ = value_.astype(float)
        support = _support_bounds(support_bounds)
        bounds = _value_bounds(value_bounds, value_.dtype)
        value_ = eqx.error_if(
            value_,
            ~jnp.isfinite(value_) | (value_ < bounds[0]) | (value_ > bounds[1]),
            "Constant property value is nonfinite or outside its declared value bounds.",
        )
        quantity_ = _identifier(quantity, "Property quantity")
        coordinate_ = _identifier(coordinate, "Property coordinate")
        value_unit_ = _identifier(value_unit, "Property value unit")
        coordinate_unit_ = _identifier(coordinate_unit, "Property coordinate unit")
        source_ = _identifier(source_id, "Property source ID")
        self.value = value_
        self.support_bounds = support
        self.value_bounds = bounds
        self.quantity = quantity_
        self.coordinate = coordinate_
        self.value_unit = value_unit_
        self.coordinate_unit = coordinate_unit_
        self.source_id = source_
        self.law_id = canonical_fingerprint(
            {
                "kind": "battery-constant-property-law",
                "quantity": quantity_,
                "coordinate": coordinate_,
                "value_unit": value_unit_,
                "coordinate_unit": coordinate_unit_,
                "source_id": source_,
            }
        )

    def evaluate(
        self, query: ArrayLike, /, *, derivative_order: int = 0
    ) -> InterpolationResult:
        if derivative_order not in (0, 1):
            raise ValueError(
                "Constant property laws support derivative orders zero and one."
            )
        query_ = jnp.asarray(query, dtype=self.support_bounds.dtype)
        support = (
            jnp.isfinite(query_)
            & (query_ >= self.support_bounds[0])
            & (query_ <= self.support_bounds[1])
        )
        value = (
            jnp.zeros_like(query_, dtype=self.value.dtype)
            if derivative_order
            else (jnp.zeros_like(query_, dtype=self.value.dtype) + self.value)
        )
        return InterpolationResult(jnp.where(support, value, 0.0), support)

    def __call__(self, query: ArrayLike, /) -> InterpolationResult:
        return self.evaluate(query)


class TabulatedPropertyLaw(StrictModule):
    """Bounded linear property table with explicit interpolation support evidence."""

    nodes: Array
    values: Array
    source_mask: Array
    value_bounds: Array
    quantity: str = eqx.field(static=True)
    coordinate: str = eqx.field(static=True)
    value_unit: str = eqx.field(static=True)
    coordinate_unit: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)

    def __init__(
        self,
        nodes: ArrayLike,
        values: ArrayLike,
        /,
        *,
        value_bounds: ArrayLike = (-jnp.inf, jnp.inf),
        source_mask: ArrayLike | None = None,
        quantity: str,
        coordinate: str,
        value_unit: str,
        coordinate_unit: str,
        source_id: str,
    ):
        nodes_ = jnp.asarray(nodes)
        values_ = jnp.asarray(values)
        if nodes_.ndim != 1 or int(nodes_.size) < 2:
            raise ValueError(
                "Tabulated property nodes must contain at least two coordinates."
            )
        if values_.shape != nodes_.shape:
            raise ValueError(
                "Tabulated scalar property values must match the node shape."
            )
        if jnp.issubdtype(nodes_.dtype, jnp.complexfloating) or jnp.issubdtype(
            values_.dtype, jnp.complexfloating
        ):
            raise TypeError("Tabulated property nodes and values must be real-valued.")
        dtype = jnp.result_type(nodes_, values_, float)
        nodes_ = nodes_.astype(dtype)
        values_ = values_.astype(dtype)
        node_host = np.asarray(nodes_, dtype=float)
        if not np.all(np.isfinite(node_host)) or np.any(np.diff(node_host) <= 0.0):
            raise ValueError(
                "Tabulated property nodes must be finite and strictly increasing."
            )
        mask = (
            jnp.ones(nodes_.shape, dtype=bool)
            if source_mask is None
            else jnp.asarray(source_mask, dtype=bool)
        )
        if mask.shape != nodes_.shape:
            raise ValueError("Tabulated property source_mask must match the node shape.")
        mask_host = np.asarray(mask, dtype=bool)
        if np.count_nonzero(mask_host) < 2:
            raise ValueError(
                "Tabulated property support requires at least two active nodes."
            )
        bounds = _value_bounds(value_bounds, dtype)
        values_ = eqx.error_if(
            values_,
            jnp.any(
                mask
                & (~jnp.isfinite(values_) | (values_ < bounds[0]) | (values_ > bounds[1]))
            ),
            "Active property values are nonfinite or outside declared value bounds.",
        )
        quantity_ = _identifier(quantity, "Property quantity")
        coordinate_ = _identifier(coordinate, "Property coordinate")
        value_unit_ = _identifier(value_unit, "Property value unit")
        coordinate_unit_ = _identifier(coordinate_unit, "Property coordinate unit")
        source_ = _identifier(source_id, "Property source ID")
        self.nodes = nodes_
        self.values = values_
        self.source_mask = mask
        self.value_bounds = bounds
        self.quantity = quantity_
        self.coordinate = coordinate_
        self.value_unit = value_unit_
        self.coordinate_unit = coordinate_unit_
        self.source_id = source_
        self.law_id = canonical_fingerprint(
            {
                "kind": "battery-tabulated-property-law",
                "quantity": quantity_,
                "coordinate": coordinate_,
                "value_unit": value_unit_,
                "coordinate_unit": coordinate_unit_,
                "source_id": source_,
                "node_count": int(nodes_.size),
                "mask": mask_host.astype(int).tolist(),
            }
        )

    @property
    def support_bounds(self) -> Array:
        active = jnp.nonzero(self.source_mask, size=self.source_mask.size, fill_value=-1)[
            0
        ]
        count = jnp.sum(self.source_mask)
        first = active[0]
        last = active[jnp.maximum(count - 1, 0)]
        return jnp.stack((self.nodes[first], self.nodes[last]))

    def evaluate(
        self, query: ArrayLike, /, *, derivative_order: int = 0
    ) -> InterpolationResult:
        if derivative_order not in (0, 1):
            raise ValueError(
                "Linear property laws support derivative orders zero and one."
            )
        return linear_interpolate(
            self.nodes,
            self.values,
            query,
            derivative_order=derivative_order,
            bounds="fill",
            source_mask=self.source_mask,
            mask_mode="strict",
            fill_value=0.0,
        )

    def __call__(self, query: ArrayLike, /) -> InterpolationResult:
        return self.evaluate(query)


class ConcentrationTemperaturePropertyLaw(StrictModule):
    """Bounded bilinear electrolyte property on concentration-temperature axes."""

    concentration_nodes_mol_m3: Array
    temperature_nodes_k: Array
    values: Array
    source_mask: Array
    value_bounds: Array
    quantity: str = eqx.field(static=True)
    value_unit: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)

    def __init__(
        self,
        concentration_nodes_mol_m3: ArrayLike,
        temperature_nodes_k: ArrayLike,
        values: ArrayLike,
        /,
        *,
        value_bounds: ArrayLike = (-jnp.inf, jnp.inf),
        source_mask: ArrayLike | None = None,
        quantity: str,
        value_unit: str,
        source_id: str,
    ):
        concentration_nodes = jnp.asarray(concentration_nodes_mol_m3)
        temperature_nodes = jnp.asarray(temperature_nodes_k)
        table = jnp.asarray(values)
        if concentration_nodes.ndim != 1 or int(concentration_nodes.size) < 2:
            raise ValueError(
                "Concentration-temperature laws require at least two concentration nodes."
            )
        if temperature_nodes.ndim != 1 or int(temperature_nodes.size) < 2:
            raise ValueError(
                "Concentration-temperature laws require at least two temperature nodes."
            )
        expected_shape = (
            int(concentration_nodes.size),
            int(temperature_nodes.size),
        )
        if table.shape != expected_shape:
            raise ValueError(
                "Concentration-temperature property values must match both axes."
            )
        if (
            jnp.issubdtype(concentration_nodes.dtype, jnp.complexfloating)
            or jnp.issubdtype(temperature_nodes.dtype, jnp.complexfloating)
            or jnp.issubdtype(table.dtype, jnp.complexfloating)
        ):
            raise TypeError(
                "Concentration-temperature property data must be real-valued."
            )
        dtype = jnp.result_type(
            concentration_nodes,
            temperature_nodes,
            table,
            float,
        )
        concentration_nodes = concentration_nodes.astype(dtype)
        temperature_nodes = temperature_nodes.astype(dtype)
        table = table.astype(dtype)
        concentration_host = np.asarray(concentration_nodes, dtype=float)
        temperature_host = np.asarray(temperature_nodes, dtype=float)
        if (
            np.any(~np.isfinite(concentration_host))
            or np.any(np.diff(concentration_host) <= 0.0)
            or concentration_host[0] <= 0.0
        ):
            raise ValueError(
                "Electrolyte concentration nodes must be finite, positive, and "
                "strictly increasing."
            )
        if (
            np.any(~np.isfinite(temperature_host))
            or np.any(np.diff(temperature_host) <= 0.0)
            or temperature_host[0] <= 0.0
        ):
            raise ValueError(
                "Temperature nodes must be finite, positive, and strictly increasing."
            )
        mask = (
            jnp.ones(expected_shape, dtype=bool)
            if source_mask is None
            else jnp.asarray(source_mask, dtype=bool)
        )
        if mask.shape != expected_shape:
            raise ValueError(
                "Concentration-temperature source_mask must match the value table."
            )
        mask_host = np.asarray(mask, dtype=bool)
        active_cells = (
            mask_host[:-1, :-1]
            & mask_host[1:, :-1]
            & mask_host[:-1, 1:]
            & mask_host[1:, 1:]
        )
        if not np.any(active_cells):
            raise ValueError(
                "Concentration-temperature support requires one complete active cell."
            )
        bounds = _value_bounds(value_bounds, dtype)
        table = eqx.error_if(
            table,
            jnp.any(
                mask & (~jnp.isfinite(table) | (table < bounds[0]) | (table > bounds[1]))
            ),
            "Active bivariate property values are nonfinite or outside declared "
            "value bounds.",
        )
        quantity_ = _identifier(quantity, "Property quantity")
        value_unit_ = _identifier(value_unit, "Property value unit")
        source_ = _identifier(source_id, "Property source ID")
        self.concentration_nodes_mol_m3 = concentration_nodes
        self.temperature_nodes_k = temperature_nodes
        self.values = table
        self.source_mask = mask
        self.value_bounds = bounds
        self.quantity = quantity_
        self.value_unit = value_unit_
        self.source_id = source_
        self.law_id = canonical_fingerprint(
            {
                "kind": "battery-concentration-temperature-property-law",
                "quantity": quantity_,
                "value_unit": value_unit_,
                "source_id": source_,
                "concentration_node_count": expected_shape[0],
                "temperature_node_count": expected_shape[1],
                "mask": mask_host.astype(int).tolist(),
            }
        )

    @property
    def support_bounds(self) -> Array:
        return jnp.stack(
            (
                jnp.stack(
                    (
                        self.concentration_nodes_mol_m3[0],
                        self.concentration_nodes_mol_m3[-1],
                    )
                ),
                jnp.stack((self.temperature_nodes_k[0], self.temperature_nodes_k[-1])),
            )
        )

    def evaluate(
        self,
        concentration_mol_m3: ArrayLike,
        temperature_k: ArrayLike,
        /,
    ) -> InterpolationResult:
        concentration, temperature = jnp.broadcast_arrays(
            jnp.asarray(concentration_mol_m3),
            jnp.asarray(temperature_k),
        )
        coordinates = jnp.stack((concentration, temperature), axis=-1)
        stencil = rectilinear_stencil(
            (self.concentration_nodes_mol_m3, self.temperature_nodes_k),
            coordinates,
            boundary=("constant", "constant"),
        )
        return apply_gather_stencil(
            self.values.reshape((-1,)),
            stencil,
            source_mask=self.source_mask.reshape((-1,)),
            mask_mode="strict",
        )

    def __call__(
        self,
        concentration_mol_m3: ArrayLike,
        temperature_k: ArrayLike,
        /,
    ) -> InterpolationResult:
        return self.evaluate(concentration_mol_m3, temperature_k)


__all__ = [
    "ConcentrationTemperaturePropertyLaw",
    "ConstantPropertyLaw",
    "TabulatedPropertyLaw",
]
