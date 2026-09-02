#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import product

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import ArraySpace, FunctionLinearOperator
from ..iga._overlay import IntegrationOverlay
from ..iga._topology import PatchAtlas
from ._interface import (
    assemble_contact_interface_traction,
    ContactInterfaceKinematics,
    ContactInterfacePlan,
    evaluate_contact_interface,
)
from ._precision import ContactPrecisionPolicy
from ._proxy import ContactProxyPlan, PreparedContactProxy
from ._surface import CollisionSurfacePlan


class IGATraceProjectionEvidence(StrictModule):
    primal_pairing: Array
    transpose_pairing: Array
    transpose_residual: Array
    scale: Array
    finite: Array
    successful: Array
    projection_id: str = eqx.field(static=True)


class IGATraceProjection(StrictModule, NonTrainableState):
    """Arbitrary coefficient-to-trace map with an explicitly supplied exact transpose."""

    matrix: Array
    constant_reproduction_error: Array
    source_coefficient_count: int = eqx.field(static=True)
    trace_coefficient_count: int = eqx.field(static=True)
    projection_id: str = eqx.field(static=True)

    def __init__(
        self,
        matrix: ArrayLike,
        /,
        *,
        require_constant_reproduction: bool = True,
        constant_tolerance: float = 1.0e-12,
        projection_id: str | None = None,
    ):
        values = np.asarray(matrix, dtype=float)
        tolerance = float(constant_tolerance)
        if values.ndim != 2 or 0 in values.shape:
            raise ValueError("IGA trace projection must be one nonempty matrix.")
        if np.any(~np.isfinite(values)):
            raise ValueError("IGA trace projection must be finite.")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("constant_tolerance must be finite and positive.")
        constant_error = np.max(np.abs(values.sum(axis=1) - 1.0), initial=0.0)
        if require_constant_reproduction and constant_error > tolerance:
            raise ValueError(
                "IGA geometric trace projection must reproduce constants exactly."
            )
        self.matrix = jnp.asarray(values)
        self.constant_reproduction_error = jnp.asarray(constant_error)
        self.source_coefficient_count = int(values.shape[1])
        self.trace_coefficient_count = int(values.shape[0])
        self.projection_id = (
            canonical_fingerprint(
                {
                    "kind": "iga-trace-projection",
                    "matrix": array_tree_fingerprint(values),
                    "constant_tolerance": tolerance.hex(),
                }
            )
            if projection_id is None
            else str(projection_id)
        )
        if not self.projection_id:
            raise ValueError("projection_id must be nonempty.")

    def apply(self, coefficients: ArrayLike, /) -> Array:
        values = jnp.asarray(coefficients)
        if values.ndim == 0 or values.shape[0] != self.source_coefficient_count:
            raise ValueError("IGA trace source has an incompatible leading dimension.")
        return ein.contract("ts,s...->t...", self.matrix.astype(values.dtype), values)

    def transpose(self, trace_dual: ArrayLike, /) -> Array:
        values = jnp.asarray(trace_dual)
        if values.ndim == 0 or values.shape[0] != self.trace_coefficient_count:
            raise ValueError("IGA trace dual has an incompatible leading dimension.")
        return ein.contract("ts,t...->s...", self.matrix.astype(values.dtype), values)

    def duality_evidence(
        self,
        coefficients: ArrayLike,
        trace_dual: ArrayLike,
        /,
    ) -> IGATraceProjectionEvidence:
        source = jnp.asarray(coefficients)
        dual = jnp.asarray(trace_dual, dtype=source.dtype)
        mapped = self.apply(source)
        pulled = self.transpose(dual)
        if mapped.shape != dual.shape or pulled.shape != source.shape:
            raise ValueError("IGA trace duality operands have incompatible shapes.")
        primal = jnp.sum(mapped * dual)
        transpose = jnp.sum(source * pulled)
        residual = primal - transpose
        scale = jnp.maximum(1.0, jnp.maximum(jnp.abs(primal), jnp.abs(transpose)))
        tolerance = jnp.finfo(source.dtype).eps * max(32, 4 * self.matrix.size)
        finite = jnp.all(jnp.isfinite(jnp.stack((primal, transpose, residual, scale))))
        successful = finite & (jnp.abs(residual) <= tolerance * scale)
        return IGATraceProjectionEvidence(
            primal,
            transpose,
            residual,
            scale,
            finite,
            successful,
            self.projection_id,
        )


class IGASweptPatchBounds(StrictModule, NonTrainableState):
    lower: Array
    upper: Array
    inflation: Array
    patch_ids: tuple[str, ...] = eqx.field(static=True)
    geometry_certificate_id: str = eqx.field(static=True)
    finite: Array
    conservative: Array
    successful: Array
    epoch_id: str = eqx.field(static=True)


class PreparedIGASplinePatchProxy(StrictModule, NonTrainableState):
    proxy: PreparedContactProxy
    patch_ids: tuple[str, ...] = eqx.field(static=True)
    geometry_certificate_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def positions(self, control_displacement: ArrayLike, /) -> Array:
        return self.proxy.positions(control_displacement)

    def pullback(self, proxy_dual: ArrayLike, /) -> Array:
        return self.proxy.pullback(proxy_dual)


class CertifiedSplinePatchProxyPlan(StrictModule, NonTrainableState):
    """Candidate-only spline proxy whose error is never promoted to geometry truth."""

    atlas: PatchAtlas
    topology: CollisionSurfacePlan
    projection: IGATraceProjection
    patch_control_indices: Array
    proxy_vertex_patch_indices: Array
    patch_approximation_error: Array
    convex_hull_certified: bool = eqx.field(static=True)
    geometry_certificate_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        atlas: PatchAtlas,
        topology: CollisionSurfacePlan,
        projection: IGATraceProjection,
        patch_control_indices: ArrayLike,
        proxy_vertex_patch_indices: ArrayLike,
        patch_approximation_error: ArrayLike,
        /,
        *,
        convex_hull_certified: bool,
        geometry_certificate_id: str,
    ):
        if not isinstance(atlas, PatchAtlas):
            raise TypeError("atlas must be PatchAtlas.")
        if not isinstance(topology, CollisionSurfacePlan):
            raise TypeError("topology must be CollisionSurfacePlan.")
        if not isinstance(projection, IGATraceProjection):
            raise TypeError("projection must be IGATraceProjection.")
        controls = np.asarray(patch_control_indices)
        vertex_patch = np.asarray(proxy_vertex_patch_indices)
        error = np.asarray(patch_approximation_error, dtype=float)
        patch_count = len(atlas.patches)
        if (
            controls.ndim != 2
            or controls.shape[0] != patch_count
            or not np.issubdtype(controls.dtype, np.integer)
        ):
            raise TypeError(
                "patch_control_indices must be one padded integer row per atlas patch."
            )
        controls = controls.astype(np.int32, copy=False)
        if np.any(controls < -1) or np.any(
            controls >= projection.source_coefficient_count
        ):
            raise ValueError("IGA proxy patch control index is invalid.")
        if np.any(np.all(controls < 0, axis=1)):
            raise ValueError(
                "Every IGA proxy patch needs at least one control coefficient."
            )
        for row in controls:
            active = row[row >= 0]
            if np.unique(active).size != active.size:
                raise ValueError(
                    "IGA proxy patch control indices must be unique per patch."
                )
        if vertex_patch.shape != (topology.vertex_count,) or not np.issubdtype(
            vertex_patch.dtype, np.integer
        ):
            raise TypeError("proxy_vertex_patch_indices must map every proxy vertex.")
        vertex_patch = vertex_patch.astype(np.int32, copy=False)
        if np.any(vertex_patch < 0) or np.any(vertex_patch >= patch_count):
            raise ValueError("IGA proxy vertex references an unknown patch.")
        if projection.trace_coefficient_count != topology.vertex_count:
            raise ValueError(
                "IGA proxy projection rows must match the collision proxy vertices."
            )
        if error.shape == ():
            error = np.full((patch_count,), float(error), dtype=float)
        if (
            error.shape != (patch_count,)
            or np.any(~np.isfinite(error))
            or np.any(error < 0.0)
        ):
            raise ValueError(
                "IGA patch approximation error must be finite and nonnegative per patch."
            )
        certificate = str(geometry_certificate_id)
        if not certificate:
            raise ValueError("geometry_certificate_id must be nonempty.")
        self.atlas = atlas
        self.topology = topology
        self.projection = projection
        self.patch_control_indices = jnp.asarray(controls)
        self.proxy_vertex_patch_indices = jnp.asarray(vertex_patch)
        self.patch_approximation_error = jnp.asarray(error)
        self.convex_hull_certified = bool(convex_hull_certified)
        self.geometry_certificate_id = certificate
        self.plan_id = canonical_fingerprint(
            {
                "kind": "certified-spline-patch-proxy-plan",
                "atlas": atlas.atlas_id,
                "topology": topology.topology_id,
                "projection": projection.projection_id,
                "patch_controls": array_tree_fingerprint(controls),
                "vertex_patch": array_tree_fingerprint(vertex_patch),
                "error": array_tree_fingerprint(error),
                "convex_hull_certified": bool(convex_hull_certified),
                "geometry_certificate": certificate,
            }
        )

    @property
    def control_coefficient_count(self) -> int:
        return self.projection.source_coefficient_count

    def prepare(
        self,
        rest_control_positions: ArrayLike,
        /,
        *,
        precision: ContactPrecisionPolicy | None = None,
    ) -> PreparedIGASplinePatchProxy:
        if not self.convex_hull_certified:
            raise ValueError(
                "IGA contact proxy preparation requires a convex-hull geometry certificate."
            )
        rest = np.asarray(rest_control_positions)
        expected = (self.control_coefficient_count, self.topology.ambient_dimension)
        if rest.shape != expected or np.any(~np.isfinite(rest)):
            raise ValueError("IGA proxy rest control positions are invalid.")
        source = ArraySpace(expected, dtype=rest.dtype)
        target = ArraySpace(
            (self.topology.vertex_count, self.topology.ambient_dimension),
            dtype=rest.dtype,
        )
        matrix = self.projection.matrix

        def map_controls(value):
            return ein.contract("vc,cd->vd", matrix.astype(value.dtype), value)

        def pull_proxy(value):
            return ein.contract("vc,vd->cd", matrix.astype(value.dtype), value)

        operator = FunctionLinearOperator(
            map_controls,
            source=source,
            target=target,
            transpose_action=pull_proxy,
            closure_convert=False,
            operator_id=canonical_fingerprint(
                {
                    "kind": "iga-proxy-control-map",
                    "plan": self.plan_id,
                }
            ),
        )
        rest_proxy = np.asarray(self.projection.apply(rest))
        vertex_error = np.asarray(self.patch_approximation_error)[
            np.asarray(self.proxy_vertex_patch_indices)
        ]
        proxy = ContactProxyPlan(
            self.topology,
            vertex_error,
            certified=True,
        ).prepare(rest_proxy, operator, precision=precision)
        if not bool(proxy.evidence.successful):
            raise ValueError(
                "Certified IGA contact proxy preparation failed evidence checks."
            )
        return PreparedIGASplinePatchProxy(
            proxy,
            tuple(patch.patch_id for patch in self.atlas.patches),
            self.geometry_certificate_id,
            self.plan_id,
        )

    def swept_bounds(
        self,
        start_control_positions: ArrayLike,
        end_control_positions: ArrayLike,
        /,
        *,
        activation_distance: float = 0.0,
    ) -> IGASweptPatchBounds:
        start = jnp.asarray(start_control_positions)
        end = jnp.asarray(end_control_positions, dtype=start.dtype)
        expected = (self.control_coefficient_count, self.topology.ambient_dimension)
        if start.shape != expected or end.shape != expected:
            raise ValueError("IGA swept-bound control positions changed shape.")
        activation = float(activation_distance)
        if not np.isfinite(activation) or activation < 0.0:
            raise ValueError("activation_distance must be finite and nonnegative.")
        safe = jnp.clip(self.patch_control_indices, 0, self.control_coefficient_count - 1)
        mask = self.patch_control_indices >= 0
        start_patch = start[safe]
        end_patch = end[safe]
        lower_values = jnp.minimum(start_patch, end_patch)
        upper_values = jnp.maximum(start_patch, end_patch)
        lower = jnp.min(jnp.where(mask[..., None], lower_values, jnp.inf), axis=1)
        upper = jnp.max(jnp.where(mask[..., None], upper_values, -jnp.inf), axis=1)
        inflation = self.patch_approximation_error.astype(start.dtype) + activation
        lower = jnp.nextafter(lower - inflation[:, None], -jnp.inf)
        upper = jnp.nextafter(upper + inflation[:, None], jnp.inf)
        finite = jnp.all(jnp.isfinite(start)) & jnp.all(jnp.isfinite(end))
        conservative = jnp.asarray(self.convex_hull_certified)
        successful = finite & conservative & jnp.all(lower <= upper)
        return IGASweptPatchBounds(
            lower,
            upper,
            inflation,
            tuple(patch.patch_id for patch in self.atlas.patches),
            self.geometry_certificate_id,
            finite,
            conservative,
            successful,
            canonical_fingerprint(
                {
                    "kind": "iga-swept-patch-bounds",
                    "plan": self.plan_id,
                    "activation_distance": activation.hex(),
                    "shape": expected,
                }
            ),
        )


class IGACommonRefinementEvidence(StrictModule):
    plus_partition_error: Array
    minus_partition_error: Array
    minimum_quadrature_weight: Array
    covered_measure: Array
    finite: Array
    coverage_certified: Array
    exact_transpose: Array
    successful: Array
    overlay_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class IGAMortarResidual(StrictModule):
    plus_trace_residual: Array
    minus_trace_residual: Array
    plus_control_residual: Array
    minus_control_residual: Array
    action_reaction_residual: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class IGAMortarDualityEvidence(StrictModule):
    primal_pairing: Array
    transpose_pairing: Array
    transpose_residual: Array
    scale: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


def _dense_trace_routes(matrix: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
    supports = tuple(np.flatnonzero(row != 0.0) for row in matrix)
    width = max((values.size for values in supports), default=0)
    if width == 0:
        raise ValueError("Every IGA mortar quadrature row needs trace support.")
    indices = np.zeros((matrix.shape[0], width), dtype=np.int32)
    weights = np.zeros((matrix.shape[0], width), dtype=float)
    for row, active in enumerate(supports):
        indices[row, : active.size] = active
        weights[row, : active.size] = matrix[row, active]
    return indices, weights


def _bspline_basis_matrix(
    knots: ArrayLike,
    degree: int,
    points: np.ndarray,
    /,
) -> np.ndarray:
    knot = np.asarray(knots, dtype=float)
    p = int(degree)
    if (
        knot.ndim != 1
        or p < 1
        or knot.size < 2 * (p + 1)
        or np.any(~np.isfinite(knot))
        or np.any(np.diff(knot) < 0.0)
    ):
        raise ValueError("IGA mortar knot vector/degree is invalid.")
    lower = knot[p]
    upper = knot[-p - 1]
    if not lower < upper or np.any(points < lower) or np.any(points > upper):
        raise ValueError("IGA mortar quadrature leaves the active knot domain.")
    basis = (
        (points[:, None] >= knot[:-1][None, :]) & (points[:, None] < knot[1:][None, :])
    ).astype(float)
    endpoint = points == upper
    for order in range(1, p + 1):
        width = knot.size - order - 1
        updated = np.zeros((points.size, width), dtype=float)
        for index in range(width):
            left_denominator = knot[index + order] - knot[index]
            right_denominator = knot[index + order + 1] - knot[index + 1]
            if left_denominator > 0.0:
                updated[:, index] += (
                    (points - knot[index]) / left_denominator * basis[:, index]
                )
            if right_denominator > 0.0:
                updated[:, index] += (
                    (knot[index + order + 1] - points)
                    / right_denominator
                    * basis[:, index + 1]
                )
        basis = updated
    if np.any(endpoint):
        basis[endpoint] = 0.0
        basis[endpoint, -1] = 1.0
    return basis


def _tensor_basis_matrix(
    knots: Sequence[ArrayLike],
    degrees: Sequence[int],
    points: np.ndarray,
    rational_weights: ArrayLike | None,
    /,
) -> np.ndarray:
    axes = tuple(
        _bspline_basis_matrix(knot, degree, points[:, axis])
        for axis, (knot, degree) in enumerate(zip(knots, degrees, strict=True))
    )
    result = axes[0]
    for axis in axes[1:]:
        result = (result[:, :, None] * axis[:, None, :]).reshape((points.shape[0], -1))
    if rational_weights is None:
        return result
    weights = np.asarray(rational_weights, dtype=float).reshape((-1,))
    if (
        weights.shape != (result.shape[1],)
        or np.any(~np.isfinite(weights))
        or np.any(weights <= 0.0)
    ):
        raise ValueError("IGA mortar rational weights must be finite and positive.")
    numerator = result * weights[None, :]
    denominator = numerator.sum(axis=1)
    if np.any(~np.isfinite(denominator)) or np.any(denominator <= 0.0):
        raise ValueError("IGA mortar rational denominator is not positive.")
    return numerator / denominator[:, None]


def _common_quadrature(
    axis_breaks: Sequence[ArrayLike],
    quadrature_order: int,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    breaks = tuple(np.asarray(axis, dtype=float) for axis in axis_breaks)
    order = int(quadrature_order)
    if order <= 0 or not breaks or len(breaks) > 2:
        raise ValueError(
            "IGA mortar supports positive-order curve or surface quadrature."
        )
    for axis in breaks:
        if (
            axis.ndim != 1
            or axis.size < 2
            or np.any(~np.isfinite(axis))
            or np.any(np.diff(axis) <= 0.0)
        ):
            raise ValueError(
                "IGA common-refinement breaks must be finite and increasing."
            )
    roots, weights = np.polynomial.legendre.leggauss(order)
    points: list[tuple[float, ...]] = []
    quadrature_weights: list[float] = []
    cells: list[int] = []
    bounds: list[tuple[tuple[float, float], ...]] = []
    cell_shape = tuple(axis.size - 1 for axis in breaks)
    for cell_multi in product(*(range(size) for size in cell_shape)):
        intervals = tuple(
            (breaks[axis][cell], breaks[axis][cell + 1])
            for axis, cell in enumerate(cell_multi)
        )
        cell_id = int(np.ravel_multi_index(cell_multi, cell_shape))
        bounds.append(intervals)
        for point_multi in product(range(order), repeat=len(breaks)):
            coordinate = []
            measure = 1.0
            for axis, root_index in enumerate(point_multi):
                lower, upper = intervals[axis]
                coordinate.append(
                    0.5 * ((upper - lower) * roots[root_index] + lower + upper)
                )
                measure *= 0.5 * (upper - lower) * weights[root_index]
            points.append(tuple(coordinate))
            quadrature_weights.append(measure)
            cells.append(cell_id)
    return (
        np.asarray(points, dtype=float),
        np.asarray(quadrature_weights, dtype=float),
        np.asarray(cells, dtype=np.int32),
        np.asarray(bounds, dtype=float),
    )


class IGACommonRefinementMortarPlan(StrictModule, NonTrainableState):
    """Exact common-refinement quadrature between arbitrary spline trace projections."""

    overlay: IntegrationOverlay
    plus_projection: IGATraceProjection
    minus_projection: IGATraceProjection
    interface: ContactInterfacePlan
    parameter_points: Array
    cell_indices: Array
    cell_parameter_bounds: Array
    evidence: IGACommonRefinementEvidence
    plus_participant: str = eqx.field(static=True)
    minus_participant: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        overlay: IntegrationOverlay,
        plus_projection: IGATraceProjection,
        minus_projection: IGATraceProjection,
        plus_basis: ArrayLike,
        minus_basis: ArrayLike,
        parameter_points: ArrayLike,
        cell_indices: ArrayLike,
        cell_parameter_bounds: ArrayLike,
        reference_normal: ArrayLike,
        quadrature_weight: ArrayLike,
        /,
        *,
        plus_participant: str,
        minus_participant: str,
        coverage_certified: bool,
    ):
        if not isinstance(overlay, IntegrationOverlay):
            raise TypeError("overlay must be IntegrationOverlay.")
        if not isinstance(plus_projection, IGATraceProjection) or not isinstance(
            minus_projection, IGATraceProjection
        ):
            raise TypeError("IGA mortar traces require typed trace projections.")
        plus_name = str(plus_participant)
        minus_name = str(minus_participant)
        if (
            not plus_name
            or not minus_name
            or plus_name == minus_name
            or plus_name not in overlay.participant_names
            or minus_name not in overlay.participant_names
        ):
            raise ValueError(
                "IGA mortar participants must be distinct overlay participants."
            )
        plus = np.asarray(plus_basis, dtype=float)
        minus = np.asarray(minus_basis, dtype=float)
        points = np.asarray(parameter_points, dtype=float)
        cells = np.asarray(cell_indices)
        bounds = np.asarray(cell_parameter_bounds, dtype=float)
        normal = np.asarray(reference_normal, dtype=float)
        measure = np.asarray(quadrature_weight, dtype=float)
        if plus.ndim != 2 or minus.ndim != 2 or plus.shape[0] != minus.shape[0]:
            raise ValueError("IGA mortar basis evaluations must be matching matrices.")
        capacity = plus.shape[0]
        if (
            plus.shape[1] != plus_projection.trace_coefficient_count
            or minus.shape[1] != minus_projection.trace_coefficient_count
        ):
            raise ValueError("IGA mortar basis width does not match trace projection.")
        if (
            points.ndim != 2
            or points.shape[0] != capacity
            or points.shape[1] not in (1, 2)
        ):
            raise ValueError("IGA mortar parameter points require curve/surface layout.")
        if cells.shape != (capacity,) or not np.issubdtype(cells.dtype, np.integer):
            raise TypeError("IGA mortar cell indices must be one integer vector.")
        if bounds.ndim != 3 or bounds.shape[1:] != (points.shape[1], 2):
            raise ValueError("IGA mortar cell parameter bounds are invalid.")
        if np.any(cells < 0) or np.any(cells >= bounds.shape[0]):
            raise ValueError("IGA mortar quadrature references an unknown overlay cell.")
        if normal.ndim == 1:
            normal = np.broadcast_to(normal, (capacity, normal.size)).copy()
        if normal.shape[0] != capacity or normal.shape[1] not in (2, 3):
            raise ValueError("IGA mortar normals require capacity by ambient dimension.")
        if measure.shape != (capacity,):
            raise ValueError("IGA mortar quadrature weight has invalid shape.")
        if (
            np.any(~np.isfinite(plus))
            or np.any(~np.isfinite(minus))
            or np.any(~np.isfinite(points))
            or np.any(~np.isfinite(bounds))
            or np.any(~np.isfinite(normal))
            or np.any(~np.isfinite(measure))
            or np.any(measure <= 0.0)
        ):
            raise ValueError("IGA mortar quadrature data must be finite and positive.")
        plus_error = np.max(np.abs(plus.sum(axis=1) - 1.0), initial=0.0)
        minus_error = np.max(np.abs(minus.sum(axis=1) - 1.0), initial=0.0)
        tolerance = 256.0 * np.finfo(float).eps * max(1, plus.shape[1], minus.shape[1])
        if plus_error > tolerance or minus_error > tolerance:
            raise ValueError("IGA mortar basis must reproduce constants.")
        plus_indices, plus_weights = _dense_trace_routes(plus)
        minus_indices, minus_weights = _dense_trace_routes(minus)
        route_keys = np.arange(capacity, dtype=np.int64)
        interface = ContactInterfacePlan(
            plus_indices,
            plus_weights,
            minus_indices,
            minus_weights,
            normal,
            measure,
            plus_node_count=plus.shape[1],
            minus_node_count=minus.shape[1],
            route_keys=route_keys,
        )
        plan_id = canonical_fingerprint(
            {
                "kind": "iga-common-refinement-mortar-plan",
                "overlay": overlay.overlay_id,
                "plus_projection": plus_projection.projection_id,
                "minus_projection": minus_projection.projection_id,
                "plus_participant": plus_name,
                "minus_participant": minus_name,
                "plus_basis": array_tree_fingerprint(plus),
                "minus_basis": array_tree_fingerprint(minus),
                "points": array_tree_fingerprint(points),
                "cells": array_tree_fingerprint(cells),
                "bounds": array_tree_fingerprint(bounds),
                "normal": array_tree_fingerprint(normal),
                "measure": array_tree_fingerprint(measure),
                "coverage_certified": bool(coverage_certified),
            }
        )
        finite = np.isfinite(plus_error) and np.isfinite(minus_error)
        successful = finite and bool(coverage_certified)
        evidence = IGACommonRefinementEvidence(
            jnp.asarray(plus_error),
            jnp.asarray(minus_error),
            jnp.asarray(np.min(measure)),
            jnp.asarray(np.sum(measure)),
            jnp.asarray(finite),
            jnp.asarray(bool(coverage_certified)),
            jnp.asarray(True),
            jnp.asarray(successful),
            overlay.overlay_id,
            plan_id,
        )
        self.overlay = overlay
        self.plus_projection = plus_projection
        self.minus_projection = minus_projection
        self.interface = interface
        self.parameter_points = jnp.asarray(points)
        self.cell_indices = jnp.asarray(cells, dtype=jnp.int32)
        self.cell_parameter_bounds = jnp.asarray(bounds)
        self.evidence = evidence
        self.plus_participant = plus_name
        self.minus_participant = minus_name
        self.plan_id = plan_id

    @classmethod
    def from_bspline_traces(
        cls,
        overlay: IntegrationOverlay,
        plus_projection: IGATraceProjection,
        minus_projection: IGATraceProjection,
        plus_knots: Sequence[ArrayLike],
        minus_knots: Sequence[ArrayLike],
        plus_degrees: Sequence[int],
        minus_degrees: Sequence[int],
        reference_normal: ArrayLike | Callable[[np.ndarray], ArrayLike],
        surface_jacobian: ArrayLike | Callable[[np.ndarray], ArrayLike],
        /,
        *,
        plus_participant: str,
        minus_participant: str,
        quadrature_order: int,
        plus_rational_weights: ArrayLike | None = None,
        minus_rational_weights: ArrayLike | None = None,
        coverage_certified: bool,
    ) -> IGACommonRefinementMortarPlan:
        if not isinstance(overlay, IntegrationOverlay):
            raise TypeError("overlay must be IntegrationOverlay.")
        plus_knot_values = tuple(plus_knots)
        minus_knot_values = tuple(minus_knots)
        plus_degree_values = tuple(int(value) for value in plus_degrees)
        minus_degree_values = tuple(int(value) for value in minus_degrees)
        dimension = len(overlay.axis_breaks)
        if (
            dimension not in (1, 2)
            or len(plus_knot_values) != dimension
            or len(minus_knot_values) != dimension
            or len(plus_degree_values) != dimension
            or len(minus_degree_values) != dimension
        ):
            raise ValueError("IGA mortar trace dimensions must match the overlay.")
        points, reference_weight, cells, bounds = _common_quadrature(
            overlay.axis_breaks, quadrature_order
        )
        plus_basis = _tensor_basis_matrix(
            plus_knot_values,
            plus_degree_values,
            points,
            plus_rational_weights,
        )
        minus_basis = _tensor_basis_matrix(
            minus_knot_values,
            minus_degree_values,
            points,
            minus_rational_weights,
        )
        normal = (
            np.asarray(reference_normal(points), dtype=float)
            if callable(reference_normal)
            else np.asarray(reference_normal, dtype=float)
        )
        jacobian = (
            np.asarray(surface_jacobian(points), dtype=float)
            if callable(surface_jacobian)
            else np.asarray(surface_jacobian, dtype=float)
        )
        if jacobian.shape == ():
            jacobian = np.full((points.shape[0],), float(jacobian), dtype=float)
        if (
            jacobian.shape != (points.shape[0],)
            or np.any(~np.isfinite(jacobian))
            or np.any(jacobian <= 0.0)
        ):
            raise ValueError("IGA mortar surface Jacobian must be finite and positive.")
        return cls(
            overlay,
            plus_projection,
            minus_projection,
            plus_basis,
            minus_basis,
            points,
            cells,
            bounds,
            normal,
            reference_weight * jacobian,
            plus_participant=plus_participant,
            minus_participant=minus_participant,
            coverage_certified=coverage_certified,
        )

    @property
    def capacity(self) -> int:
        return self.interface.capacity

    @property
    def ambient_dimension(self) -> int:
        return self.interface.ambient_dimension

    def evaluate(
        self,
        plus_control_positions: ArrayLike,
        minus_control_positions: ArrayLike,
        /,
    ) -> ContactInterfaceKinematics:
        if not bool(self.evidence.successful):
            raise ValueError(
                "IGA mortar evaluation requires certified common refinement."
            )
        plus_trace = self.plus_projection.apply(plus_control_positions)
        minus_trace = self.minus_projection.apply(minus_control_positions)
        return evaluate_contact_interface(self.interface, plus_trace, minus_trace)

    def assemble(self, traction: ArrayLike, /) -> IGAMortarResidual:
        if not bool(self.evidence.successful):
            raise ValueError("IGA mortar assembly requires certified common refinement.")
        residual = assemble_contact_interface_traction(self.interface, traction)
        plus_control = self.plus_projection.transpose(residual.plus_residual)
        minus_control = self.minus_projection.transpose(residual.minus_residual)
        finite = (
            residual.finite
            & jnp.all(jnp.isfinite(plus_control))
            & jnp.all(jnp.isfinite(minus_control))
        )
        return IGAMortarResidual(
            residual.plus_residual,
            residual.minus_residual,
            plus_control,
            minus_control,
            residual.action_reaction_residual,
            finite,
            finite & residual.successful,
            self.plan_id,
        )

    def duality_evidence(
        self,
        plus_control_variation: ArrayLike,
        minus_control_variation: ArrayLike,
        traction: ArrayLike,
        /,
    ) -> IGAMortarDualityEvidence:
        plus = jnp.asarray(plus_control_variation)
        minus = jnp.asarray(minus_control_variation, dtype=plus.dtype)
        traction_ = jnp.asarray(traction, dtype=plus.dtype)
        if traction_.shape != (self.capacity, self.ambient_dimension):
            raise ValueError("IGA mortar traction has invalid shape.")
        plus_trace = self.plus_projection.apply(plus)
        minus_trace = self.minus_projection.apply(minus)
        kinematic = evaluate_contact_interface(self.interface, plus_trace, minus_trace)
        primal = jnp.sum(
            kinematic.relative_displacement
            * traction_
            * self.interface.quadrature_weight[:, None].astype(traction_.dtype)
        )
        residual = self.assemble(traction_)
        transpose = jnp.sum(plus * residual.plus_control_residual) + jnp.sum(
            minus * residual.minus_control_residual
        )
        defect = primal - transpose
        scale = jnp.maximum(1.0, jnp.maximum(jnp.abs(primal), jnp.abs(transpose)))
        tolerance = jnp.finfo(plus.dtype).eps * max(
            64,
            8
            * self.capacity
            * self.ambient_dimension
            * (
                self.plus_projection.source_coefficient_count
                + self.minus_projection.source_coefficient_count
            ),
        )
        finite = jnp.all(jnp.isfinite(jnp.stack((primal, transpose, defect, scale))))
        successful = finite & (jnp.abs(defect) <= tolerance * scale)
        return IGAMortarDualityEvidence(
            primal,
            transpose,
            defect,
            scale,
            finite,
            successful,
            self.plan_id,
        )


__all__ = [
    "CertifiedSplinePatchProxyPlan",
    "IGACommonRefinementEvidence",
    "IGACommonRefinementMortarPlan",
    "IGAMortarDualityEvidence",
    "IGAMortarResidual",
    "IGASweptPatchBounds",
    "IGATraceProjection",
    "IGATraceProjectionEvidence",
    "PreparedIGASplinePatchProxy",
]
