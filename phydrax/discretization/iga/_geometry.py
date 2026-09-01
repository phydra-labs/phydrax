#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._basis import TensorSplineBasisSpec


class NURBSGeometryState(StrictModule):
    """Differentiable NURBS values with a unique mean-one positive weight gauge."""

    control_points: Array
    weights: Array

    def __init__(self, control_points: ArrayLike, weights: ArrayLike, /):
        points = jnp.asarray(control_points)
        weights_ = jnp.asarray(weights)
        if points.ndim < 2:
            raise ValueError(
                "NURBS control points require tensor control axes and one coordinate axis."
            )
        if weights_.shape != points.shape[:-1]:
            raise ValueError("NURBS weights must match the control-point tensor shape.")
        if points.shape[-1] <= 0:
            raise ValueError("NURBS control points require a positive ambient dimension.")
        if jnp.issubdtype(points.dtype, jnp.complexfloating) or jnp.issubdtype(
            weights_.dtype, jnp.complexfloating
        ):
            raise TypeError("NURBS geometry values must be real.")
        dtype = jnp.result_type(points, weights_, float)
        points = points.astype(dtype)
        weights_ = weights_.astype(dtype)
        points = eqx.error_if(
            points,
            jnp.any(~jnp.isfinite(points)),
            "NURBS control points must be finite.",
        )
        weights_ = eqx.error_if(
            weights_,
            jnp.any(~jnp.isfinite(weights_)) | jnp.any(weights_ <= 0.0),
            "NURBS weights must be finite and strictly positive.",
        )
        gauge = jnp.mean(weights_)
        self.control_points = points
        self.weights = weights_ / gauge

    @property
    def ambient_dimension(self) -> int:
        return int(self.control_points.shape[-1])

    @property
    def control_shape(self) -> tuple[int, ...]:
        return tuple(int(size) for size in self.weights.shape)


class IsogeometricRuntimeData(StrictModule):
    """Sole numeric-version owner for a fixed IGA topology and geometry layout."""

    control_points: Array
    weights: Array
    topology_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)
    numeric_version: str = eqx.field(static=True)
    runtime_id: str = eqx.field(static=True)

    def __init__(
        self,
        basis: TensorSplineBasisSpec,
        geometry: NURBSGeometryState,
        /,
        *,
        topology_id: str,
        numeric_version: str,
    ):
        if not isinstance(basis, TensorSplineBasisSpec):
            raise TypeError("basis must be a TensorSplineBasisSpec.")
        if not isinstance(geometry, NURBSGeometryState):
            raise TypeError("geometry must be a NURBSGeometryState.")
        if geometry.control_shape != basis.control_shape:
            raise ValueError(
                "NURBS geometry control shape must exactly match the spline basis."
            )
        topology = str(topology_id)
        version = str(numeric_version)
        if not topology or not version:
            raise ValueError("IGA topology and numeric version must be non-empty.")
        geometry_layout = canonical_fingerprint(
            {
                "kind": "isogeometric-geometry-layout",
                "basis_layout": basis.layout_id,
                "control_shape": list(basis.control_shape),
                "ambient_dimension": geometry.ambient_dimension,
            }
        )
        self.control_points = geometry.control_points
        self.weights = geometry.weights
        self.topology_id = topology
        self.geometry_layout_id = geometry_layout
        self.numeric_version = version
        self.runtime_id = canonical_fingerprint(
            {
                "kind": "isogeometric-runtime",
                "topology": topology,
                "geometry_layout": geometry_layout,
                "numeric_version": version,
            }
        )

    @property
    def geometry(self) -> NURBSGeometryState:
        return NURBSGeometryState(self.control_points, self.weights)


class IsogeometricGeometryEvidence(StrictModule, NonTrainableState):
    """Dimensionless scale-aware evidence for a realized NURBS geometry."""

    coordinate_scale: Array
    minimum_weight_ratio: Array
    minimum_rank_ratio: Array
    minimum_orientation_ratio: Array
    ambient_dimension: int = eqx.field(static=True)
    parametric_dimension: int = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        coordinate_scale: ArrayLike,
        minimum_weight_ratio: ArrayLike,
        minimum_rank_ratio: ArrayLike,
        minimum_orientation_ratio: ArrayLike,
        /,
        *,
        ambient_dimension: int,
        parametric_dimension: int,
        evidence_id: str,
    ):
        ambient = int(ambient_dimension)
        parametric = int(parametric_dimension)
        identifier = str(evidence_id)
        if ambient < parametric or parametric <= 0 or not identifier:
            raise ValueError("IGA geometry evidence dimensions and ID must be valid.")
        self.coordinate_scale = jnp.asarray(coordinate_scale)
        self.minimum_weight_ratio = jnp.asarray(minimum_weight_ratio)
        self.minimum_rank_ratio = jnp.asarray(minimum_rank_ratio)
        self.minimum_orientation_ratio = jnp.asarray(minimum_orientation_ratio)
        self.ambient_dimension = ambient
        self.parametric_dimension = parametric
        self.evidence_id = identifier


class IsogeometricH1QualificationPolicy(StrictModule, NonTrainableState):
    """Frozen checked S1 geometry, convergence, parity, and sensitivity gates."""

    weight_tolerance: float | None = eqx.field(static=True)
    rank_tolerance: float | None = eqx.field(static=True)
    orientation_tolerance: float | None = eqx.field(static=True)
    refinement_levels: int = eqx.field(static=True)
    h1_rate_slack: float = eqx.field(static=True)
    l2_rate_slack: float = eqx.field(static=True)
    residual_factor: float = eqx.field(static=True)
    residual_epsilon_factor: float = eqx.field(static=True)
    parity_factor: float = eqx.field(static=True)
    parity_epsilon_factor: float = eqx.field(static=True)
    duality_factor: float = eqx.field(static=True)
    duality_epsilon_factor: float = eqx.field(static=True)
    taylor_slope_min: float = eqx.field(static=True)
    taylor_slope_max: float = eqx.field(static=True)
    taylor_minimum_intervals: int = eqx.field(static=True)
    taylor_step_count: int = eqx.field(static=True)
    quadrature_error_fraction: float = eqx.field(static=True)
    quadrature_reference_increment: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        weight_tolerance: float | None = None,
        rank_tolerance: float | None = None,
        orientation_tolerance: float | None = None,
        refinement_levels: int = 4,
        h1_rate_slack: float = 0.25,
        l2_rate_slack: float = 0.25,
        residual_factor: float = 10.0,
        residual_epsilon_factor: float = 1024.0,
        parity_factor: float = 0.01,
        parity_epsilon_factor: float = 1024.0,
        duality_factor: float = 0.01,
        duality_epsilon_factor: float = 4096.0,
        taylor_slope_min: float = 1.8,
        taylor_slope_max: float = 2.2,
        taylor_minimum_intervals: int = 3,
        taylor_step_count: int = 6,
        quadrature_error_fraction: float = 0.1,
        quadrature_reference_increment: int = 2,
    ):
        def optional_tolerance(name: str, value: float | None) -> float | None:
            if value is None:
                return None
            result = float(value)
            if not isfinite(result) or result <= 0.0:
                raise ValueError(f"{name} must be finite and positive or None.")
            return result

        weight = optional_tolerance("weight_tolerance", weight_tolerance)
        rank = optional_tolerance("rank_tolerance", rank_tolerance)
        orientation = optional_tolerance("orientation_tolerance", orientation_tolerance)
        levels = int(refinement_levels)
        h1_slack = float(h1_rate_slack)
        l2_slack = float(l2_rate_slack)
        residual = float(residual_factor)
        residual_epsilon = float(residual_epsilon_factor)
        parity = float(parity_factor)
        parity_epsilon = float(parity_epsilon_factor)
        duality = float(duality_factor)
        duality_epsilon = float(duality_epsilon_factor)
        slope_min = float(taylor_slope_min)
        slope_max = float(taylor_slope_max)
        minimum_intervals = int(taylor_minimum_intervals)
        step_count = int(taylor_step_count)
        quadrature_fraction = float(quadrature_error_fraction)
        quadrature_increment = int(quadrature_reference_increment)
        real_values = (
            h1_slack,
            l2_slack,
            residual,
            residual_epsilon,
            parity,
            parity_epsilon,
            duality,
            duality_epsilon,
            slope_min,
            slope_max,
            quadrature_fraction,
        )
        if not all(isfinite(value) for value in real_values):
            raise ValueError("IGA qualification thresholds must be finite.")
        if levels < 4:
            raise ValueError(
                "IGA convergence qualification requires at least four levels."
            )
        if h1_slack < 0.0 or l2_slack < 0.0:
            raise ValueError("IGA convergence rate slacks must be non-negative.")
        if (
            min(residual, residual_epsilon, parity, parity_epsilon) <= 0.0
            or min(duality, duality_epsilon) <= 0.0
        ):
            raise ValueError(
                "IGA residual, parity, and duality factors must be positive."
            )
        if not 0.0 < slope_min < slope_max or step_count < 4:
            raise ValueError("IGA Taylor slope interval/count is invalid.")
        if minimum_intervals < 3 or minimum_intervals >= step_count:
            raise ValueError("IGA Taylor minimum interval count is invalid.")
        if not 0.0 < quadrature_fraction <= 1.0 or quadrature_increment <= 0:
            raise ValueError("IGA quadrature qualification thresholds are invalid.")
        self.weight_tolerance = weight
        self.rank_tolerance = rank
        self.orientation_tolerance = orientation
        self.refinement_levels = levels
        self.h1_rate_slack = h1_slack
        self.l2_rate_slack = l2_slack
        self.residual_factor = residual
        self.residual_epsilon_factor = residual_epsilon
        self.parity_factor = parity
        self.parity_epsilon_factor = parity_epsilon
        self.duality_factor = duality
        self.duality_epsilon_factor = duality_epsilon
        self.taylor_slope_min = slope_min
        self.taylor_slope_max = slope_max
        self.taylor_minimum_intervals = minimum_intervals
        self.taylor_step_count = step_count
        self.quadrature_error_fraction = quadrature_fraction
        self.quadrature_reference_increment = quadrature_increment
        self.policy_id = canonical_fingerprint(
            {
                "kind": "isogeometric-h1-qualification-policy",
                "weight_tolerance": weight,
                "rank_tolerance": rank,
                "orientation_tolerance": orientation,
                "default_geometry_tolerance": "sqrt-epsilon",
                "refinement_levels": levels,
                "h1_rate_slack": h1_slack,
                "l2_rate_slack": l2_slack,
                "residual_factor": residual,
                "residual_epsilon_factor": residual_epsilon,
                "parity_factor": parity,
                "parity_epsilon_factor": parity_epsilon,
                "duality_factor": duality,
                "duality_epsilon_factor": duality_epsilon,
                "taylor_slope_min": slope_min,
                "taylor_slope_max": slope_max,
                "taylor_minimum_intervals": minimum_intervals,
                "taylor_step_count": step_count,
                "quadrature_error_fraction": quadrature_fraction,
                "quadrature_reference_increment": quadrature_increment,
            }
        )

    def geometry_tolerances(self, dtype, /) -> tuple[Array, Array, Array]:
        default = jnp.sqrt(jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype))
        return (
            default
            if self.weight_tolerance is None
            else jnp.asarray(self.weight_tolerance, dtype=dtype),
            default
            if self.rank_tolerance is None
            else jnp.asarray(self.rank_tolerance, dtype=dtype),
            default
            if self.orientation_tolerance is None
            else jnp.asarray(self.orientation_tolerance, dtype=dtype),
        )

    def check(
        self, evidence: IsogeometricGeometryEvidence, /
    ) -> IsogeometricGeometryEvidence:
        if not isinstance(evidence, IsogeometricGeometryEvidence):
            raise TypeError("evidence must be IsogeometricGeometryEvidence.")
        weight_tolerance, rank_tolerance, orientation_tolerance = (
            self.geometry_tolerances(evidence.minimum_rank_ratio.dtype)
        )
        weight_ratio = eqx.error_if(
            evidence.minimum_weight_ratio,
            ~jnp.isfinite(evidence.minimum_weight_ratio)
            | (evidence.minimum_weight_ratio <= weight_tolerance),
            "IGA rational denominator W failed the scale-aware positivity check.",
        )
        rank_ratio = eqx.error_if(
            evidence.minimum_rank_ratio,
            ~jnp.isfinite(evidence.minimum_rank_ratio)
            | (evidence.minimum_rank_ratio <= rank_tolerance),
            "IGA geometry Jacobian failed the scale-aware rank check.",
        )
        orientation_ratio = eqx.error_if(
            evidence.minimum_orientation_ratio,
            ~jnp.isfinite(evidence.minimum_orientation_ratio)
            | (evidence.minimum_orientation_ratio <= orientation_tolerance),
            "IGA geometry Jacobian failed the orientation check.",
        )
        return eqx.tree_at(
            lambda value: (
                value.minimum_weight_ratio,
                value.minimum_rank_ratio,
                value.minimum_orientation_ratio,
            ),
            evidence,
            (weight_ratio, rank_ratio, orientation_ratio),
        )


__all__ = [
    "IsogeometricGeometryEvidence",
    "IsogeometricH1QualificationPolicy",
    "IsogeometricRuntimeData",
    "NURBSGeometryState",
]
