#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jaxtyping import Array

import phydrax.ein as ein

from .._strict import StrictModule
from ..metrix import RiemannianMetric
from ._cochain import CochainComplexIR
from ._continuous_bridge import ContinuousCochainBridge, OrientedCellParameterization


class MetricCochainAssembly(StrictModule):
    """A cochain complex with metric-derived primal, dual, and Hodge measures."""

    complex: CochainComplexIR
    primal_measures: tuple[Array, ...]
    dual_measures: tuple[Array, ...]
    hodge_stars: tuple[Array, ...]
    valid: Array
    finite: Array
    minimum_primal_measure: Array
    minimum_dual_measure: Array
    minimum_hodge_star: Array

    def __init__(
        self,
        complex: CochainComplexIR,
        primal_measures: tuple[Array, ...],
        dual_measures: tuple[Array, ...],
        hodge_stars: tuple[Array, ...],
        /,
    ):
        self.complex = complex
        self.primal_measures = primal_measures
        self.dual_measures = dual_measures
        self.hodge_stars = hodge_stars
        primal = jnp.concatenate(primal_measures)
        dual = jnp.concatenate(dual_measures)
        stars = jnp.concatenate(hodge_stars)
        self.finite = (
            jnp.all(jnp.isfinite(primal))
            & jnp.all(jnp.isfinite(dual))
            & jnp.all(jnp.isfinite(stars))
        )
        self.minimum_primal_measure = jnp.min(primal)
        self.minimum_dual_measure = jnp.min(dual)
        self.minimum_hodge_star = jnp.min(stars)
        self.valid = (
            self.finite
            & (self.minimum_primal_measure > 0.0)
            & (self.minimum_dual_measure > 0.0)
            & (self.minimum_hodge_star > 0.0)
        )


def _parameterized_measures(
    parameterization: OrientedCellParameterization,
    metric: RiemannianMetric,
    /,
) -> Array:
    if bool(jnp.any(parameterization.quadrature_weights < 0)):
        raise ValueError("Metric cell assembly requires nonnegative quadrature weights.")

    def cell_measure(cell: Array) -> Array:
        def density(reference: Array) -> Array:
            point = parameterization.map_function(cell, reference)
            jacobian = jnp.asarray(parameterization.jacobian_function(cell, reference))
            expected = (parameterization.ambient_dimension, parameterization.degree)
            if jacobian.shape != expected:
                raise ValueError(
                    f"Cell Jacobian must have shape {expected}; got {jacobian.shape}."
                )
            if parameterization.degree == 0:
                return jnp.asarray(1.0, dtype=point.dtype)
            induced = ein.contract("ai,ab,bj->ij", jacobian, metric(point), jacobian)
            return jnp.sqrt(jnp.linalg.det(induced))

        values = jax.vmap(density)(parameterization.reference_points)
        return jnp.sum(parameterization.quadrature_weights * values)

    cells = jnp.arange(parameterization.cell_count, dtype=jnp.int32)
    return jax.vmap(cell_measure)(cells)


def assemble_metric_cochain_complex(
    bridge: ContinuousCochainBridge,
    metric: RiemannianMetric,
    dual_parameterizations: Sequence[OrientedCellParameterization],
    /,
) -> MetricCochainAssembly:
    """Assemble diagonal Hodge stars from paired primal and dual cells."""
    if not isinstance(bridge, ContinuousCochainBridge):
        raise TypeError("bridge must be a ContinuousCochainBridge.")
    if not isinstance(metric, RiemannianMetric):
        raise TypeError("metric must be a RiemannianMetric.")
    if not bridge.chart.compatible_with(metric.chart):
        raise ValueError("Bridge and metric charts must match.")
    dual = tuple(dual_parameterizations)
    if len(dual) != len(bridge.parameterizations):
        raise ValueError("One dual parameterization is required for every degree.")
    ambient_dimension = bridge.chart.dimension
    primal_measures = []
    dual_measures = []
    for degree, (primal_parameterization, dual_parameterization) in enumerate(
        zip(bridge.parameterizations, dual, strict=True)
    ):
        if not isinstance(dual_parameterization, OrientedCellParameterization):
            raise TypeError("Dual cells must be OrientedCellParameterization objects.")
        if (
            dual_parameterization.degree != ambient_dimension - degree
            or dual_parameterization.cell_count != primal_parameterization.cell_count
            or dual_parameterization.ambient_dimension != ambient_dimension
        ):
            raise ValueError(
                "Dual cell degree, count, and ambient dimension are incompatible."
            )
        primal_measures.append(_parameterized_measures(primal_parameterization, metric))
        dual_measures.append(_parameterized_measures(dual_parameterization, metric))
    primal_tuple = tuple(primal_measures)
    dual_tuple = tuple(dual_measures)
    stars = tuple(
        dual_measure / primal_measure
        for primal_measure, dual_measure in zip(primal_tuple, dual_tuple, strict=True)
    )
    complex = CochainComplexIR(
        bridge.complex.cell_counts,
        bridge.complex.incidences,
        stars,
        primal_measures=primal_tuple,
        dual_measures=dual_tuple,
        boundary_masks=bridge.complex.boundary_masks,
        coordinates=bridge.complex.coordinates,
        harmonic_subspace=None,
        validate=True,
    )
    return MetricCochainAssembly(complex, primal_tuple, dual_tuple, stars)


__all__ = ["MetricCochainAssembly", "assemble_metric_cochain_complex"]
