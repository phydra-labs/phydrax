#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..ein import contract
from ..linalg import AbstractLinearOperator


class GraphMetricPrior(StrictModule, NonTrainableState):
    """Volume-weighted damping and edge-metric H1 prior on scalar cell values."""

    cell_measures: Array
    edge_owner: Array
    edge_neighbour: Array
    edge_weights: Array
    damping_precision: float = eqx.field(static=True)
    gradient_precision: float = eqx.field(static=True)
    reference: Array
    parameter_unit_id: str = eqx.field(static=True)
    prior_id: str = eqx.field(static=True)

    def __init__(
        self,
        cell_measures: ArrayLike,
        edge_owner: ArrayLike,
        edge_neighbour: ArrayLike,
        edge_weights: ArrayLike,
        /,
        *,
        damping_precision: float = 0.0,
        gradient_precision: float = 1.0,
        reference: ArrayLike = 0.0,
        parameter_unit_id: str = "dimensionless",
    ):
        measures = np.asarray(cell_measures, dtype=float)
        owner, neighbour = np.asarray(edge_owner), np.asarray(edge_neighbour)
        weights = np.asarray(edge_weights, dtype=float)
        if (
            measures.ndim != 1
            or measures.size == 0
            or np.any(~np.isfinite(measures))
            or np.any(measures <= 0)
        ):
            raise ValueError("Prior cell measures must be a positive finite vector.")
        if (
            owner.ndim != 1
            or neighbour.shape != owner.shape
            or weights.shape != owner.shape
            or not np.issubdtype(owner.dtype, np.integer)
            or not np.issubdtype(neighbour.dtype, np.integer)
            or np.any(owner < 0)
            or np.any(neighbour < 0)
            or np.any(owner >= measures.size)
            or np.any(neighbour >= measures.size)
            or np.any(owner == neighbour)
            or np.any(~np.isfinite(weights))
            or np.any(weights <= 0)
        ):
            raise ValueError("Prior edges and metric weights are invalid.")
        damping, gradient = float(damping_precision), float(gradient_precision)
        if (
            not np.isfinite(damping)
            or damping < 0
            or not np.isfinite(gradient)
            or gradient < 0
        ):
            raise ValueError("Prior precisions must be finite and nonnegative.")
        if damping == 0 and gradient == 0:
            raise ValueError("At least one prior precision must be positive.")
        reference_ = np.broadcast_to(np.asarray(reference, dtype=float), measures.shape)
        if np.any(~np.isfinite(reference_)):
            raise ValueError("Prior reference must be finite and cell-sized.")
        unit = str(parameter_unit_id).strip()
        if not unit:
            raise ValueError("Prior parameter unit identity is required.")
        self.cell_measures = jnp.asarray(measures)
        self.edge_owner = jnp.asarray(owner, dtype=jnp.int32)
        self.edge_neighbour = jnp.asarray(neighbour, dtype=jnp.int32)
        self.edge_weights = jnp.asarray(weights)
        self.damping_precision, self.gradient_precision = damping, gradient
        self.reference = jnp.asarray(reference_)
        self.parameter_unit_id = unit
        self.prior_id = canonical_fingerprint(
            {
                "kind": "graph-metric-prior",
                "cell_measures": measures,
                "edges": (owner, neighbour),
                "edge_weights": weights,
                "precisions": (damping, gradient),
                "reference": reference_,
                "parameter_unit_id": unit,
            }
        )

    def components(self, values: ArrayLike, /) -> tuple[Array, Array]:
        field = jnp.asarray(values)
        if field.shape != self.cell_measures.shape:
            raise ValueError("Prior field must match cell measures.")
        if jnp.iscomplexobj(field):
            raise TypeError("Graph-metric prior requires a real field.")
        field = eqx.error_if(
            field,
            jnp.any(~jnp.isfinite(field)),
            "Graph-metric prior field must be finite.",
        )
        centered = field - self.reference
        damping = self.damping_precision * jnp.sum(self.cell_measures * centered**2)
        difference = field[self.edge_neighbour] - field[self.edge_owner]
        gradient = self.gradient_precision * jnp.sum(self.edge_weights * difference**2)
        return damping, gradient

    def log_prob(self, values: ArrayLike, /) -> Array:
        damping, gradient = self.components(values)
        return -0.5 * (damping + gradient)


class TotalVariationPrior(StrictModule, NonTrainableState):
    edge_owner: Array
    edge_neighbour: Array
    edge_weights: Array
    scale: float = eqx.field(static=True)
    smoothing: float = eqx.field(static=True)
    differentiability: Literal["nonsmooth", "smoothed"] = eqx.field(static=True)
    field_count: int = eqx.field(static=True)

    def __init__(
        self,
        edge_owner: ArrayLike,
        edge_neighbour: ArrayLike,
        edge_weights: ArrayLike,
        scale: float,
        /,
        *,
        smoothing: float = 0.0,
    ):
        owner, neighbour = np.asarray(edge_owner), np.asarray(edge_neighbour)
        weights = np.asarray(edge_weights, dtype=float)
        scale_, smoothing_ = float(scale), float(smoothing)
        if (
            owner.ndim != 1
            or owner.size == 0
            or neighbour.shape != owner.shape
            or weights.shape != owner.shape
            or not np.issubdtype(owner.dtype, np.integer)
            or not np.issubdtype(neighbour.dtype, np.integer)
            or np.any(owner < 0)
            or np.any(neighbour < 0)
            or np.any(owner == neighbour)
            or np.any(~np.isfinite(weights))
            or np.any(weights <= 0)
            or not np.isfinite(scale_)
            or scale_ <= 0
            or not np.isfinite(smoothing_)
            or smoothing_ < 0
        ):
            raise ValueError("Total-variation graph, scale, or smoothing is invalid.")
        self.edge_owner = jnp.asarray(owner, dtype=jnp.int32)
        self.edge_neighbour = jnp.asarray(neighbour, dtype=jnp.int32)
        self.edge_weights = jnp.asarray(weights)
        self.scale, self.smoothing = scale_, smoothing_
        self.field_count = int(max(np.max(owner), np.max(neighbour))) + 1
        self.differentiability = "nonsmooth" if smoothing_ == 0 else "smoothed"

    def log_prob(self, values: ArrayLike, /) -> Array:
        field = jnp.asarray(values)
        if field.shape != (self.field_count,):
            raise ValueError("Total-variation field does not cover its graph.")
        if jnp.iscomplexobj(field):
            raise TypeError("Total-variation prior requires a real field.")
        field = eqx.error_if(
            field,
            jnp.any(~jnp.isfinite(field)),
            "Total-variation prior field must be finite.",
        )
        difference = field[self.edge_neighbour] - field[self.edge_owner]
        magnitude = (
            jnp.abs(difference)
            if self.smoothing == 0
            else jnp.sqrt(difference**2 + self.smoothing**2) - self.smoothing
        )
        return -jnp.sum(self.edge_weights * magnitude) / self.scale


class TemporalDifferencePrior(StrictModule, NonTrainableState):
    times: Array
    scale: float = eqx.field(static=True)
    order: Literal[1, 2] = eqx.field(static=True)
    time_unit_id: str = eqx.field(static=True)

    def __init__(
        self,
        times: ArrayLike,
        scale: float,
        /,
        *,
        order: Literal[1, 2] = 1,
        time_unit_id: str = "s",
    ):
        values = np.asarray(times, dtype=float)
        scale_ = float(scale)
        if (
            values.ndim != 1
            or values.size < order + 1
            or np.any(~np.isfinite(values))
            or np.any(np.diff(values) <= 0)
            or not np.isfinite(scale_)
            or scale_ <= 0
            or order not in (1, 2)
        ):
            raise ValueError(
                "Temporal prior needs ordered times, valid order, and positive scale."
            )
        unit = str(time_unit_id).strip()
        if not unit:
            raise ValueError("Temporal prior time unit identity is required.")
        self.times, self.scale, self.order, self.time_unit_id = (
            jnp.asarray(values),
            scale_,
            order,
            unit,
        )

    def standardized_difference(self, values: ArrayLike, /) -> Array:
        field = jnp.asarray(values)
        if field.ndim == 0 or field.shape[0] != self.times.size:
            raise ValueError("Temporal field leading axis must match prior times.")
        if jnp.iscomplexobj(field):
            raise TypeError("Temporal-difference prior requires a real field.")
        field = eqx.error_if(
            field,
            jnp.any(~jnp.isfinite(field)),
            "Temporal-difference prior field must be finite.",
        )
        dt = jnp.diff(self.times)
        first = jnp.diff(field, axis=0) / dt.reshape((-1,) + (1,) * (field.ndim - 1))
        if self.order == 1:
            return first / self.scale
        midpoint_dt = 0.5 * (dt[1:] + dt[:-1])
        second = jnp.diff(first, axis=0) / midpoint_dt.reshape(
            (-1,) + (1,) * (field.ndim - 1)
        )
        return second / self.scale

    def log_prob(self, values: ArrayLike, /) -> Array:
        difference = self.standardized_difference(values)
        return -0.5 * jnp.real(jnp.vdot(difference, difference))


class CrossGradientPrior(StrictModule, NonTrainableState):
    gradient_matrix: Array
    cell_measures: Array
    dimension: int = eqx.field(static=True)
    scale: float = eqx.field(static=True)
    normalized: bool = eqx.field(static=True)
    epsilon: float = eqx.field(static=True)

    def __init__(
        self,
        gradient_matrix: ArrayLike,
        cell_measures: ArrayLike,
        dimension: Literal[2, 3],
        scale: float,
        /,
        *,
        normalized: bool = False,
        epsilon: float = 0.0,
    ):
        matrix = np.asarray(gradient_matrix, dtype=float)
        measures = np.asarray(cell_measures, dtype=float)
        dimension_, scale_, epsilon_ = int(dimension), float(scale), float(epsilon)
        if (
            matrix.ndim != 3
            or matrix.shape[0] != measures.size
            or matrix.shape[1] != dimension_
            or dimension_ not in (2, 3)
            or np.any(~np.isfinite(matrix))
            or np.any(~np.isfinite(measures))
            or np.any(measures <= 0)
            or not np.isfinite(scale_)
            or scale_ <= 0
            or not np.isfinite(epsilon_)
            or epsilon_ < 0
            or (normalized and epsilon_ <= 0)
        ):
            raise ValueError(
                "Cross-gradient operator, measures, scale, or normalization is invalid."
            )
        self.gradient_matrix = jnp.asarray(matrix)
        self.cell_measures = jnp.asarray(measures)
        self.dimension, self.scale = dimension_, scale_
        self.normalized, self.epsilon = bool(normalized), epsilon_

    def log_prob(self, first: ArrayLike, second: ArrayLike, /) -> Array:
        first_, second_ = jnp.asarray(first), jnp.asarray(second)
        parameter_count = self.gradient_matrix.shape[2]
        if first_.shape != (parameter_count,) or second_.shape != first_.shape:
            raise ValueError(
                "Cross-gradient fields must match gradient operator columns."
            )
        if jnp.iscomplexobj(first_) or jnp.iscomplexobj(second_):
            raise TypeError("Cross-gradient prior requires real fields.")
        first_ = eqx.error_if(
            first_,
            jnp.any(~jnp.isfinite(first_)) | jnp.any(~jnp.isfinite(second_)),
            "Cross-gradient prior fields must be finite.",
        )
        first_gradient = contract("cdp,p->cd", self.gradient_matrix, first_)
        second_gradient = contract("cdp,p->cd", self.gradient_matrix, second_)
        if self.dimension == 2:
            cross = (
                first_gradient[:, 0] * second_gradient[:, 1]
                - first_gradient[:, 1] * second_gradient[:, 0]
            )
            squared = cross**2
        else:
            cross = jnp.cross(first_gradient, second_gradient)
            squared = jnp.sum(cross**2, axis=1)
        if self.normalized:
            denominator = (jnp.sum(first_gradient**2, axis=1) + self.epsilon**2) * (
                jnp.sum(second_gradient**2, axis=1) + self.epsilon**2
            )
            squared = squared / denominator
        return -0.5 * jnp.sum(self.cell_measures * squared) / self.scale**2


class SPDEPrecisionPrior(StrictModule, NonTrainableState):
    precision: AbstractLinearOperator
    mean: Array
    logdet_precision: Array | None
    normalized: bool = eqx.field(static=True)

    def __init__(
        self,
        precision: AbstractLinearOperator,
        mean: ArrayLike,
        /,
        *,
        logdet_precision: ArrayLike | None = None,
    ):
        if not isinstance(precision, AbstractLinearOperator):
            raise TypeError("SPDE precision must be a native linear operator.")
        if (
            not precision.source.compatible(precision.target)
            or not precision.properties.certifies("self_adjoint")
            or not precision.properties.certifies("positive_definite")
        ):
            raise ValueError(
                "SPDE precision must be a certified self-adjoint positive-definite endomorphism."
            )
        mean_ = jnp.asarray(mean)
        if (
            mean_.size != precision.source.size
            or jnp.iscomplexobj(mean_)
            or bool(jnp.any(~jnp.isfinite(mean_)))
        ):
            raise ValueError("SPDE prior mean must be real, finite, and match its space.")
        logdet = None if logdet_precision is None else jnp.asarray(logdet_precision)
        if logdet is not None and (
            logdet.shape != () or jnp.iscomplexobj(logdet) or bool(~jnp.isfinite(logdet))
        ):
            raise ValueError("SPDE log determinant must be a real finite scalar or None.")
        self.precision, self.mean, self.logdet_precision = precision, mean_, logdet
        self.normalized = logdet is not None

    def log_prob(self, values: ArrayLike, /) -> Array:
        field = jnp.asarray(values)
        if field.shape != self.mean.shape:
            raise ValueError("SPDE field and mean shapes disagree.")
        if jnp.iscomplexobj(field):
            raise TypeError("SPDE precision prior requires a real field.")
        field = eqx.error_if(
            field, jnp.any(~jnp.isfinite(field)), "SPDE prior field must be finite."
        )
        centered = field - self.mean
        structured = self.precision.source.unflatten(centered.reshape(-1))
        applied = self.precision.mv(structured)
        quadratic = jnp.real(self.precision.source.inner(structured, applied))
        result = -0.5 * quadratic
        if self.logdet_precision is not None:
            result = (
                result
                + 0.5 * self.logdet_precision
                - 0.5 * centered.size * jnp.log(2 * jnp.pi)
            )
        return result


__all__ = [
    "CrossGradientPrior",
    "GraphMetricPrior",
    "SPDEPrecisionPrior",
    "TemporalDifferencePrior",
    "TotalVariationPrior",
]
