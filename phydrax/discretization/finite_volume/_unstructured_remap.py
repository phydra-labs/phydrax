#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import StrEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._numerics._compensated import compensated_sum_chunks
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import positive_finite_float
from ...geometry import _supermesh as supermesh_owner
from ...linalg import (
    adjoint,
    FunctionLinearOperator,
    SmallLinearSolvePlan,
    SmallLinearSolveResult,
    solve_small_linear,
    transpose,
)
from ...sparse import (
    EdgeRelation,
    gather_routes,
    route_reduce,
    RowRelation,
    SparseLinearMap,
)
from ...typing import checked, parse
from .. import (
    _sphere_chart_records as sphere_chart_deformation_owner,
    _surface_chart_records as surface_chart_deformation_owner,
)
from .._cell_mesh import CellMesh
from .._coordinate_budget import CoordinateEnclosureBudget
from .._spaces import DiscreteFieldSpace
from .._topology_epoch import TopologyEpoch, TopologyEpochTransition
from .._transfer import FieldTransfer, TransferGeometryBinding, TransferProperties
from ..fem import _surface_chart_transfer as surface_chart_transfer_owner
from ._unstructured import UnstructuredFiniteVolumeDiscretization


class UnstructuredRemapReport(StrictModule):
    """Certified source-content coverage and physical endpoint diagnostics.

    For ``source-physical-chart-content``, target measure defects report bounded
    physical area deformation, not a UV coverage gap. Target coverage is
    certified by the native chart certificate retained on the plan.
    """

    maximum_target_coverage_defect: Array
    maximum_source_coverage_defect: Array
    uncovered_target_measure: Array
    uncovered_source_measure: Array
    donor_excess_measure: Array
    source_measure: Array
    target_measure: Array
    coverage_complete: Array
    tolerance: Array
    maximum_target_coverage_error_bound: Array
    maximum_source_coverage_error_bound: Array
    total_target_coverage_error_bound: Array
    total_source_coverage_error_bound: Array
    measure_semantics: str = eqx.field(static=True)


def _active_mask(value: ArrayLike | None, count: int, name: str, /) -> Array:
    if value is None:
        return jnp.ones((count,), dtype=jnp.bool_)
    array = jnp.asarray(value)
    if array.shape != (count,) or array.dtype != jnp.dtype(jnp.bool_):
        raise ValueError(
            f"remap.{name} must be a Boolean array with one entry per cell; got shape {array.shape} and dtype {array.dtype}."
        )
    return array


def _volume_array(
    value: ArrayLike | None, fallback: Array, count: int, name: str, /
) -> Array:
    array = fallback if value is None else jnp.asarray(value)
    if array.shape != (count,):
        raise ValueError(
            f"remap.{name} must contain one volume per cell; got shape {array.shape}."
        )
    return array


def _mask_values(values: Array, mask: Array, name: str, /) -> Array:
    expanded = mask.reshape(mask.shape + (1,) * (values.ndim - 1))
    values = eqx.error_if(
        values,
        jnp.any((~expanded) & (values != 0.0)),
        f"{name} must be exactly zero on inactive cells.",
    )
    return values


class RemapCoverageLedger(StrictModule):
    """Signed covered-minus-cell measures and their admissible magnitudes."""

    target_defect: np.ndarray
    source_defect: np.ndarray
    target_limit: np.ndarray
    source_limit: np.ndarray


def coverage_ledger(
    target_routes: np.ndarray,
    source_indices: np.ndarray,
    measures: np.ndarray,
    source_volumes: np.ndarray,
    target_volumes: np.ndarray,
    tolerance: float,
    /,
    *,
    intersection_error_bounds: np.ndarray | None = None,
    source_error_bounds: np.ndarray | None = None,
    target_error_bounds: np.ndarray | None = None,
) -> RemapCoverageLedger:
    """Return signed covered-minus-cell measures and their admissible magnitudes.

    Full coverage is admitted when every defect magnitude is at most its limit.
    """
    target_defect = (
        np.bincount(target_routes, weights=measures, minlength=target_volumes.size)
        - target_volumes
    )
    source_defect = (
        np.bincount(source_indices, weights=measures, minlength=source_volumes.size)
        - source_volumes
    )
    target_scale = np.maximum(target_volumes, np.max(target_volumes) * 1e-14)
    source_scale = np.maximum(source_volumes, np.max(source_volumes) * 1e-14)
    errors = (
        np.zeros_like(measures)
        if intersection_error_bounds is None
        else intersection_error_bounds
    )
    target_error = np.bincount(
        target_routes, weights=errors, minlength=target_volumes.size
    )
    source_error = np.bincount(
        source_indices, weights=errors, minlength=source_volumes.size
    )
    if target_error_bounds is not None:
        target_error += target_error_bounds
    if source_error_bounds is not None:
        source_error += source_error_bounds
    return RemapCoverageLedger(
        target_defect,
        source_defect,
        tolerance * target_scale + target_error,
        tolerance * source_scale + source_error,
    )


class _RemapCarrier(StrictModule):
    offsets: np.ndarray
    indices: np.ndarray
    measures: np.ndarray
    target_routes: np.ndarray
    source_volumes: np.ndarray
    target_volumes: np.ndarray
    errors: np.ndarray
    coverage: RemapCoverageLedger


class _ChartContentsBinding(StrictModule):
    source_limit: np.ndarray


class _RemapCoveragePreparation(StrictModule):
    report: UnstructuredRemapReport


class _RemapIdentity(StrictModule):
    route_id: str = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)
    coverage_id: str = eqx.field(static=True)


def _prepare_remap_carrier(
    source: UnstructuredFiniteVolumeDiscretization,
    target: UnstructuredFiniteVolumeDiscretization,
    target_offsets: ArrayLike,
    source_indices: ArrayLike,
    intersection_measures: ArrayLike,
    intersection_error_bounds: ArrayLike | None,
    tolerance_: float,
) -> _RemapCarrier:
    offsets = np.asarray(target_offsets, dtype=np.int32)
    indices = np.asarray(source_indices, dtype=np.int32)
    measures = np.asarray(intersection_measures, dtype=np.float64)
    if offsets.shape != (target.cell_count + 1,):
        raise ValueError(
            f"remap.target_offsets must contain one CSR offset per target cell; got shape {offsets.shape}."
        )
    if (
        offsets[0] != 0
        or np.any(np.diff(offsets) < 0)
        or offsets[-1] != indices.size
        or measures.shape != indices.shape
    ):
        raise ValueError(
            "remap.csr must contain monotone offsets and aligned indices and measures; got inconsistent routes."
        )
    if np.any(indices < 0) or np.any(indices >= source.cell_count):
        raise ValueError(
            "remap.source_indices must index source cells; got out-of-range indices."
        )
    if np.any(~np.isfinite(measures)) or np.any(measures <= 0.0):
        raise ValueError(
            "remap.intersection_measures must be positive and finite; got nonpositive or nonfinite measures."
        )
    source_ids = np.asarray(source.cell_global_ids)
    target_ids = np.asarray(target.cell_global_ids)
    if source_ids.shape != (source.cell_count,) or target_ids.shape != (
        target.cell_count,
    ):
        raise ValueError(
            "remap.cell_global_ids must contain one ID per cell; got incompatible shapes."
        )
    if source_ids.dtype.kind not in "iu" or target_ids.dtype.kind not in "iu":
        raise TypeError(
            "remap.cell_global_ids must have integer dtype; got a noninteger array."
        )
    if np.any(source_ids < 0) or np.any(target_ids < 0):
        raise ValueError("remap.cell_global_ids must be nonnegative; got negative IDs.")
    if (
        np.unique(source_ids).size != source_ids.size
        or np.unique(target_ids).size != target_ids.size
    ):
        raise ValueError(
            "remap.cell_global_ids must be unique within each mesh; got duplicate IDs."
        )
    target_routes = np.repeat(
        np.arange(target.cell_count, dtype=np.int32), np.diff(offsets)
    )
    source_volumes = np.asarray(source.cell_volumes)
    target_volumes = np.asarray(target.cell_volumes)
    errors = (
        np.zeros_like(measures)
        if intersection_error_bounds is None
        else np.asarray(intersection_error_bounds, dtype=np.float64)
    )
    if (
        errors.shape != measures.shape
        or np.any(~np.isfinite(errors))
        or np.any(errors < 0)
        or np.any(errors >= measures)
    ):
        raise ValueError(
            "remap.intersection_error_bounds must certify positive overlaps; got invalid shapes, nonfinite bounds, or bounds outside the overlap measures."
        )
    coverage = coverage_ledger(
        target_routes,
        indices,
        measures,
        source_volumes,
        target_volumes,
        tolerance_,
        intersection_error_bounds=errors,
        source_error_bounds=np.asarray(source.cell_volume_error_bounds),
        target_error_bounds=np.asarray(target.cell_volume_error_bounds),
    )
    return _RemapCarrier(
        offsets,
        indices,
        measures,
        target_routes,
        source_volumes,
        target_volumes,
        errors,
        coverage,
    )


def _bind_chart_contents(
    source: UnstructuredFiniteVolumeDiscretization,
    target: UnstructuredFiniteVolumeDiscretization,
    carrier: _RemapCarrier,
    surface_chart_deformation: surface_chart_deformation_owner.PreparedSurfaceChartDeformation
    | sphere_chart_deformation_owner.PreparedSphereChartDeformation
    | None,
    surface_chart_contents: surface_chart_transfer_owner.PreparedSurfaceChartFiniteVolumeContents
    | None,
    budget: CoordinateEnclosureBudget | None,
) -> _ChartContentsBinding:
    indices, measures, target_routes = (
        carrier.indices,
        carrier.measures,
        carrier.target_routes,
    )
    source_volumes, target_volumes, errors = (
        carrier.source_volumes,
        carrier.target_volumes,
        carrier.errors,
    )
    source_limit = carrier.coverage.source_limit
    if (surface_chart_deformation is None) != (surface_chart_contents is None):
        raise ValueError(
            "remap.surface_chart must pair deformation with physical contents; got an unpaired artifact."
        )
    match surface_chart_deformation, surface_chart_contents:
        case None, None:
            return _ChartContentsBinding(source_limit)
        case (
            (
                surface_chart_deformation_owner.PreparedSurfaceChartDeformation()
                | sphere_chart_deformation_owner.PreparedSphereChartDeformation()
            ),
            surface_chart_transfer_owner.PreparedSurfaceChartFiniteVolumeContents(),
        ):
            if source.cell_geometry is None or target.cell_geometry is None:
                raise ValueError(
                    "remap.surface_chart.cell_geometry must contain both prepared coordinate maps; got a missing map."
                )
            match surface_chart_deformation:
                case sphere_chart_deformation_owner.PreparedSphereChartDeformation():
                    surface_chart_deformation.require_bound(
                        source.mesh,
                        source.cell_geometry,
                        target.mesh,
                        target.cell_geometry,
                        budget=budget,
                    )
                case surface_chart_deformation_owner.PreparedSurfaceChartDeformation():
                    surface_chart_deformation.require_bound(
                        source.mesh,
                        source.cell_geometry,
                        target.mesh,
                        target.cell_geometry,
                    )
                case _:
                    raise TypeError(
                        "remap.surface_chart_deformation must declare a prepared material chart; got an unsupported kind."
                    )
            if (
                surface_chart_contents.deformation_id
                != surface_chart_deformation.deformation_id
            ):
                raise ValueError(
                    "remap.surface_chart_contents.deformation_id must match the deformation; got a different ID."
                )
            if not np.array_equal(
                surface_chart_contents.source_volumes, source_volumes
            ) or not np.array_equal(
                surface_chart_contents.target_volumes, target_volumes
            ):
                raise ValueError(
                    "remap.surface_chart_contents.volumes must match prepared FV physical areas; got different measures."
                )
            order = np.lexsort(
                (
                    np.asarray(surface_chart_contents.source_rows),
                    np.asarray(surface_chart_contents.target_rows),
                )
            )
            if not (
                np.array_equal(
                    indices, np.asarray(surface_chart_contents.source_rows)[order]
                )
                and np.array_equal(
                    target_routes, np.asarray(surface_chart_contents.target_rows)[order]
                )
                and np.array_equal(
                    measures, np.asarray(surface_chart_contents.source_contents)[order]
                )
                and np.array_equal(
                    errors,
                    np.asarray(surface_chart_contents.source_content_errors)[order],
                )
            ):
                raise ValueError(
                    "remap.surface_chart.csr must realize the physical-content certificate; got mismatched routes, measures, or bounds."
                )
            source_limit = np.asarray(
                surface_chart_contents.source_content_defect_bounds, dtype=np.float64
            )
            if (
                source_limit.shape != source_volumes.shape
                or np.any(~np.isfinite(source_limit))
                or np.any(source_limit < 0)
            ):
                raise ValueError(
                    "remap.surface_chart_contents.source_content_defect_bounds must be finite nonnegative per-cell bounds; got invalid bounds."
                )
        case _:
            raise ValueError(
                "remap.surface_chart must pair deformation with physical contents; got an unpaired artifact."
            )
    return _ChartContentsBinding(source_limit)


def _certify_remap_coverage(
    source: UnstructuredFiniteVolumeDiscretization,
    target: UnstructuredFiniteVolumeDiscretization,
    carrier: _RemapCarrier,
    source_limit: np.ndarray,
    surface_chart_deformation: surface_chart_deformation_owner.PreparedSurfaceChartDeformation
    | sphere_chart_deformation_owner.PreparedSphereChartDeformation
    | None,
    tolerance_: float,
    require_complete: bool,
) -> _RemapCoveragePreparation:
    target_routes, indices, errors = (
        carrier.target_routes,
        carrier.indices,
        carrier.errors,
    )
    source_volumes, target_volumes = carrier.source_volumes, carrier.target_volumes
    target_defect, source_defect = (
        carrier.coverage.target_defect,
        carrier.coverage.source_defect,
    )
    target_limit = carrier.coverage.target_limit
    source_complete = np.all(np.abs(source_defect) <= source_limit)
    # Target coverage is a native UV tiling for chart deformation, not the
    # false assertion that old physical piece areas equal new physical areas.
    complete = source_complete and (
        surface_chart_deformation is not None
        or np.all(np.abs(target_defect) <= target_limit)
    )
    if require_complete and not complete:
        raise ValueError(
            "remap.coverage must completely cover source and target; got measure defects outside the certified limits."
        )
    target_errors = np.bincount(
        target_routes, weights=errors, minlength=target_volumes.size
    ) + np.asarray(target.cell_volume_error_bounds)
    source_errors = np.bincount(
        indices, weights=errors, minlength=source_volumes.size
    ) + np.asarray(source.cell_volume_error_bounds)
    if surface_chart_deformation is not None:
        source_errors = np.maximum(source_errors, source_limit)
    report = UnstructuredRemapReport(
        maximum_target_coverage_defect=jnp.asarray(np.max(np.abs(target_defect))),
        maximum_source_coverage_defect=jnp.asarray(np.max(np.abs(source_defect))),
        uncovered_target_measure=jnp.asarray(np.sum(np.maximum(-target_defect, 0.0))),
        uncovered_source_measure=jnp.asarray(np.sum(np.maximum(-source_defect, 0.0))),
        donor_excess_measure=jnp.asarray(np.sum(np.maximum(source_defect, 0.0))),
        source_measure=jnp.asarray(np.sum(source_volumes)),
        target_measure=jnp.asarray(np.sum(target_volumes)),
        coverage_complete=jnp.asarray(complete),
        tolerance=jnp.asarray(tolerance_),
        maximum_target_coverage_error_bound=jnp.asarray(np.max(target_errors)),
        maximum_source_coverage_error_bound=jnp.asarray(np.max(source_errors)),
        total_target_coverage_error_bound=jnp.asarray(np.sum(target_errors)),
        total_source_coverage_error_bound=jnp.asarray(np.sum(source_errors)),
        measure_semantics="source-physical-chart-content"
        if surface_chart_deformation is not None
        else "physical-cell-overlap",
    )
    return _RemapCoveragePreparation(report)


def _prepare_remap_identity(
    source: UnstructuredFiniteVolumeDiscretization,
    target: UnstructuredFiniteVolumeDiscretization,
    carrier: _RemapCarrier,
    route_id: str | None,
    layout_id: str | None,
    require_complete: bool,
    tolerance_: float,
) -> _RemapIdentity:
    target_routes, indices, offsets = (
        carrier.target_routes,
        carrier.indices,
        carrier.offsets,
    )
    target_defect, source_defect = (
        carrier.coverage.target_defect,
        carrier.coverage.source_defect,
    )
    route_ = (
        str(route_id)
        if route_id is not None
        else canonical_fingerprint(
            {
                "kind": "unstructured-remap-route",
                "target_routes": array_tree_fingerprint(target_routes),
                "source_indices": array_tree_fingerprint(indices),
            }
        )
    )
    layout_ = (
        str(layout_id)
        if layout_id is not None
        else canonical_fingerprint(
            {
                "kind": "unstructured-remap-layout",
                "target_offsets": array_tree_fingerprint(offsets),
                "component": "cell-content",
            }
        )
    )
    if not route_ or not layout_:
        raise ValueError(
            f"remap.identifiers must contain non-empty route_id and layout_id; got route_id={route_!r}, layout_id={layout_!r}."
        )
    coverage_id = canonical_fingerprint(
        {
            "kind": "unstructured-remap-coverage-evidence",
            "source_topology": source.topology_id,
            "target_topology": target.topology_id,
            "maximum_target_defect": float(np.max(np.abs(target_defect))),
            "maximum_source_defect": float(np.max(np.abs(source_defect))),
            "require_complete": bool(require_complete),
            "tolerance": tolerance_,
        }
    )
    return _RemapIdentity(route_, layout_, coverage_id)


class UnstructuredConservativeRemapPlan(StrictModule, NonTrainableState):
    """Prepare CSR conservative content transport between immutable epochs.

    ``physical-cell-overlap`` uses geometric intersections or nested reference tilings.
    ``source-physical-chart-content`` uses old physical density integrals over
    native UV overlap pieces and divides by new physical cell
    areas. It conserves content, not constants under area-changing deformation.
    ``apply`` consumes cell averages; ``apply_content`` consumes extensive content.
    """

    source_topology_id: str = eqx.field(static=True)
    source_geometry_id: str = eqx.field(static=True)
    target_topology_id: str = eqx.field(static=True)
    target_geometry_id: str = eqx.field(static=True)
    source_cell_global_ids: Array
    target_cell_global_ids: Array
    target_offsets: Array
    source_indices: Array
    target_routes: Array
    intersection_measures: Array
    intersection_error_bounds: Array
    source_volumes: Array
    target_volumes: Array
    source_volume_error_bounds: Array
    target_volume_error_bounds: Array
    surface_chart_deformation: (
        surface_chart_deformation_owner.PreparedSurfaceChartDeformation
        | sphere_chart_deformation_owner.PreparedSphereChartDeformation
        | None
    )
    surface_chart_contents: (
        surface_chart_transfer_owner.PreparedSurfaceChartFiniteVolumeContents | None
    )
    measure_semantics: str = eqx.field(static=True)
    method: str = eqx.field(static=True)
    provenance: str = eqx.field(static=True)
    require_complete: bool = eqx.field(static=True)
    report: UnstructuredRemapReport
    coverage_evidence_id: str = eqx.field(static=True)
    route_id: str = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        source: UnstructuredFiniteVolumeDiscretization,
        target: UnstructuredFiniteVolumeDiscretization,
        target_offsets: ArrayLike,
        source_indices: ArrayLike,
        intersection_measures: ArrayLike,
        /,
        *,
        method: str,
        provenance: str,
        tolerance: float = 1e-10,
        require_complete: bool = True,
        route_id: str | None = None,
        layout_id: str | None = None,
        intersection_error_bounds: ArrayLike | None = None,
        surface_chart_deformation: surface_chart_deformation_owner.PreparedSurfaceChartDeformation
        | sphere_chart_deformation_owner.PreparedSphereChartDeformation
        | None = None,
        surface_chart_contents: surface_chart_transfer_owner.PreparedSurfaceChartFiniteVolumeContents
        | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> None:
        method_ = str(method)
        provenance_ = str(provenance)
        if not method_ or not provenance_:
            raise ValueError(
                f"remap.metadata must contain non-empty method and provenance; got method={method_!r}, provenance={provenance_!r}."
            )
        tolerance_ = positive_finite_float(tolerance, "remap.tolerance")
        carrier = _prepare_remap_carrier(
            source,
            target,
            target_offsets,
            source_indices,
            intersection_measures,
            intersection_error_bounds,
            tolerance_,
        )
        chart_binding = _bind_chart_contents(
            source,
            target,
            carrier,
            surface_chart_deformation,
            surface_chart_contents,
            budget,
        )
        coverage = _certify_remap_coverage(
            source,
            target,
            carrier,
            chart_binding.source_limit,
            surface_chart_deformation,
            tolerance_,
            require_complete,
        )
        identity = _prepare_remap_identity(
            source, target, carrier, route_id, layout_id, require_complete, tolerance_
        )
        self.source_topology_id = source.topology_id
        self.source_geometry_id = source.geometry_id
        self.target_topology_id = target.topology_id
        self.target_geometry_id = target.geometry_id
        self.source_cell_global_ids = source.cell_global_ids
        self.target_cell_global_ids = target.cell_global_ids
        self.target_offsets = jnp.asarray(carrier.offsets)
        self.source_indices = jnp.asarray(carrier.indices)
        self.target_routes = jnp.asarray(carrier.target_routes)
        self.intersection_measures = jnp.asarray(carrier.measures)
        self.intersection_error_bounds = jnp.asarray(carrier.errors)
        self.source_volumes = source.cell_volumes
        self.target_volumes = target.cell_volumes
        self.source_volume_error_bounds = source.cell_volume_error_bounds
        self.target_volume_error_bounds = target.cell_volume_error_bounds
        self.surface_chart_deformation = surface_chart_deformation
        self.surface_chart_contents = surface_chart_contents
        self.measure_semantics = (
            "source-physical-chart-content"
            if surface_chart_deformation is not None
            else "physical-cell-overlap"
        )
        self.method = method_
        self.provenance = provenance_
        self.require_complete = bool(require_complete)
        self.report = coverage.report
        self.coverage_evidence_id = identity.coverage_id
        self.route_id = identity.route_id
        self.layout_id = identity.layout_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "unstructured-conservative-remap",
                "source_topology": source.topology_id,
                "source_geometry": source.geometry_id,
                "target_topology": target.topology_id,
                "target_geometry": target.geometry_id,
                "surface_chart_deformation": None
                if surface_chart_deformation is None
                else surface_chart_deformation.deformation_id,
                "surface_chart_contents": None
                if surface_chart_contents is None
                else surface_chart_contents.contents_id,
                "source_cell_global_ids": array_tree_fingerprint(source.cell_global_ids),
                "target_cell_global_ids": array_tree_fingerprint(target.cell_global_ids),
                "target_offsets": array_tree_fingerprint(carrier.offsets),
                "source_indices": array_tree_fingerprint(carrier.indices),
                "intersection_measures": array_tree_fingerprint(carrier.measures),
                "intersection_error_bounds": array_tree_fingerprint(carrier.errors),
                "source_volume_error_bounds": array_tree_fingerprint(
                    source.cell_volume_error_bounds
                ),
                "target_volume_error_bounds": array_tree_fingerprint(
                    target.cell_volume_error_bounds
                ),
                "method": method_,
                "provenance": provenance_,
                "require_complete": bool(require_complete),
                "coverage_evidence_id": identity.coverage_id,
                "route_id": identity.route_id,
                "layout_id": identity.layout_id,
            }
        )

    def _validate_values(self, values: ArrayLike, name: str, /) -> Array:
        value = jnp.asarray(values)
        if value.ndim == 0 or value.shape[0] != self.source_volumes.size:
            raise ValueError(
                f"remap.{name} must begin with the source cell count; got shape {value.shape}."
            )
        return value

    def apply(
        self,
        source_cell_averages: ArrayLike,
        /,
        *,
        source_active_mask: ArrayLike | None = None,
        target_active_mask: ArrayLike | None = None,
        target_volumes: ArrayLike | None = None,
    ) -> Array:
        """Transfer cell averages using volume-weighted common refinement."""
        value = self._validate_values(source_cell_averages, "Remap values")
        source_active = _active_mask(
            source_active_mask, self.source_volumes.size, "source_active_mask"
        )
        target_active = _active_mask(
            target_active_mask, self.target_volumes.size, "target_active_mask"
        )
        value = _mask_values(value, source_active, "Remap values")
        trailing = (1,) * (value.ndim - 1)
        denominator = _volume_array(
            target_volumes,
            self.target_volumes,
            self.target_volumes.size,
            "target_volumes",
        )
        denominator = eqx.error_if(
            denominator,
            jnp.any(target_active & (~jnp.isfinite(denominator) | (denominator <= 0.0))),
            "Active target remap volumes must be positive and finite.",
        )
        denominator = jnp.where(
            target_active,
            denominator,
            jnp.ones_like(denominator),
        )
        # Each route carries its normalized overlap fraction m_e / V_t, so a
        # route covering its whole target cell has weight exactly one and the
        # identity common refinement reproduces the averages bit for bit.
        weights = (
            self.intersection_measures.astype(value.dtype)
            / denominator.astype(value.dtype)[self.target_routes]
        )
        weighted = value[self.source_indices] * weights.reshape((-1,) + trailing)
        target = jnp.zeros(
            (self.target_volumes.size,) + value.shape[1:], dtype=value.dtype
        )
        target = target.at[self.target_routes].add(weighted)
        return jnp.where(
            target_active.reshape(target_active.shape + trailing),
            target,
            jnp.zeros((), dtype=target.dtype),
        )

    def apply_content(
        self,
        source_content: ArrayLike,
        /,
        *,
        source_volumes: ArrayLike | None = None,
        source_active_mask: ArrayLike | None = None,
        target_active_mask: ArrayLike | None = None,
    ) -> Array:
        """Transfer extensive content without converting it to a target average.

        Each source content is distributed in proportion to its covered geometric
        measure. Thus a fully covered map preserves the global extensive integral,
        including when source and target cell volumes differ.
        """
        content = self._validate_values(source_content, "Remap content")
        source_active = _active_mask(
            source_active_mask, self.source_volumes.size, "source_active_mask"
        )
        target_active = _active_mask(
            target_active_mask, self.target_volumes.size, "target_active_mask"
        )
        content = _mask_values(content, source_active, "Remap content")
        source_measure = _volume_array(
            source_volumes,
            self.source_volumes,
            self.source_volumes.size,
            "source_volumes",
        )
        source_measure = eqx.error_if(
            source_measure,
            jnp.any(~jnp.isfinite(source_measure) | (source_measure <= 0.0)),
            "Source remap volumes must be positive and finite.",
        )
        trailing = (1,) * (content.ndim - 1)
        density = content / source_measure.astype(content.dtype).reshape((-1,) + trailing)
        weighted = density[self.source_indices] * self.intersection_measures.astype(
            content.dtype
        ).reshape((-1,) + trailing)
        target = jnp.zeros(
            (self.target_volumes.size,) + content.shape[1:], dtype=content.dtype
        )
        target = target.at[self.target_routes].add(weighted)
        return jnp.where(
            target_active.reshape(target_active.shape + trailing),
            target,
            jnp.zeros((), dtype=target.dtype),
        )

    def apply_bounded(
        self,
        source_cell_averages: ArrayLike,
        /,
        *,
        lower: float = 0.0,
        upper: float = 1.0,
        source_active_mask: ArrayLike | None = None,
        target_active_mask: ArrayLike | None = None,
    ) -> Array:
        """Transfer a bounded scalar, failing for invalid source fractions."""
        if self.surface_chart_deformation is not None:
            raise ValueError(
                "remap.measure_semantics must support density bounds; got area-changing chart-content transport."
            )
        lower_ = float(lower)
        upper_ = float(upper)
        if not np.isfinite(lower_) or not np.isfinite(upper_) or lower_ > upper_:
            raise ValueError(
                "remap.bounds must be finite with lower <= upper; got nonfinite or inverted bounds."
            )
        source = self._validate_values(source_cell_averages, "Remap values")
        source = eqx.error_if(
            source,
            jnp.any(~jnp.isfinite(source) | (source < lower_) | (source > upper_)),
            "Bounded remap source values violate the supplied bounds.",
        )
        transferred = self.apply(
            source,
            source_active_mask=source_active_mask,
            target_active_mask=target_active_mask,
        )
        # Positive overlap weights make this a convex transfer. Clipping removes
        # roundoff excursions; the explicit check rejects values outside bounds.
        repaired = jnp.clip(transferred, lower_, upper_)
        return eqx.error_if(
            repaired,
            jnp.any(~jnp.isfinite(repaired) | (repaired < lower_) | (repaired > upper_)),
            "Bounded remap could not produce values in the requested interval.",
        )

    def apply_fixed_combinatorics(
        self,
        source_cell_averages: ArrayLike,
        intersection_measures: ArrayLike,
        source_volumes: ArrayLike,
        target_volumes: ArrayLike,
        /,
    ) -> Array:
        """Differentiable remap for one frozen common-refinement route graph."""
        source = self._validate_values(
            source_cell_averages, "Fixed-combinatorics remap values"
        )
        measures = jnp.asarray(intersection_measures, dtype=source.dtype)
        source_measure = jnp.asarray(source_volumes, dtype=source.dtype)
        target_measure = jnp.asarray(target_volumes, dtype=source.dtype)
        if measures.shape != self.intersection_measures.shape:
            raise ValueError(
                "remap.dynamic.intersection_measures must preserve overlap capacity; got a different shape."
            )
        if source_measure.shape != self.source_volumes.shape:
            raise ValueError(
                "remap.dynamic.source_volumes must preserve source capacity; got a different shape."
            )
        if target_measure.shape != self.target_volumes.shape:
            raise ValueError(
                "remap.dynamic.target_volumes must preserve target capacity; got a different shape."
            )
        measures = eqx.error_if(
            measures,
            jnp.any(~jnp.isfinite(measures) | (measures <= 0.0)),
            "Fixed-combinatorics intersections must remain positive and finite.",
        )
        source_measure = eqx.error_if(
            source_measure,
            jnp.any(~jnp.isfinite(source_measure) | (source_measure <= 0.0)),
            "Fixed-combinatorics source volumes must remain positive and finite.",
        )
        target_measure = eqx.error_if(
            target_measure,
            jnp.any(~jnp.isfinite(target_measure) | (target_measure <= 0.0)),
            "Fixed-combinatorics target volumes must remain positive and finite.",
        )
        target_coverage = (
            jnp.zeros_like(target_measure).at[self.target_routes].add(measures)
        )
        source_coverage = (
            jnp.zeros_like(source_measure).at[self.source_indices].add(measures)
        )
        tolerance = self.report.tolerance.astype(source.dtype)
        coverage_valid = jnp.all(
            jnp.abs(target_coverage - target_measure)
            <= tolerance * jnp.maximum(target_measure, 1.0)
        ) & jnp.all(
            jnp.abs(source_coverage - source_measure)
            <= tolerance * jnp.maximum(source_measure, 1.0)
        )
        source = eqx.error_if(
            source,
            ~coverage_valid,
            "Fixed-combinatorics remap lost complete geometric coverage.",
        )
        trailing = (1,) * (source.ndim - 1)
        weighted = source[self.source_indices] * measures.reshape((-1,) + trailing)
        target = (
            jnp.zeros((target_measure.size,) + source.shape[1:], dtype=source.dtype)
            .at[self.target_routes]
            .add(weighted)
        )
        return target / target_measure.reshape((-1,) + trailing)

    def conservation_defect(
        self, source_cell_averages: ArrayLike, target_cell_averages: ArrayLike, /
    ) -> Array:
        source = jnp.asarray(source_cell_averages)
        target = jnp.asarray(target_cell_averages)
        if source.ndim == 0 or source.shape[0] != self.source_volumes.size:
            raise ValueError(
                "remap.source_cell_averages must begin with the source cell count; got a different shape."
            )
        if target.ndim == 0 or target.shape[0] != self.target_volumes.size:
            raise ValueError(
                "remap.target_cell_averages must begin with the target cell count; got a different shape."
            )
        source_terms = (
            self.source_volumes.astype(source.dtype).reshape(
                (-1,) + (1,) * (source.ndim - 1)
            )
            * source
        )
        target_terms = (
            self.target_volumes.astype(target.dtype).reshape(
                (-1,) + (1,) * (target.ndim - 1)
            )
            * target
        )
        return compensated_sum_chunks(
            (target_terms, -source_terms),
            output_ndim=source.ndim - 1,
        )

    def conservation_defect_content(
        self, source_content: ArrayLike, target_content: ArrayLike, /
    ) -> Array:
        source = jnp.asarray(source_content)
        target = jnp.asarray(target_content)
        if source.ndim == 0 or source.shape[0] != self.source_volumes.size:
            raise ValueError(
                "remap.source_content must begin with the source cell count; got a different shape."
            )
        if target.ndim == 0 or target.shape[0] != self.target_volumes.size:
            raise ValueError(
                f"remap.target_content must begin with the target cell count; got shape {target.shape}."
            )
        return compensated_sum_chunks(
            (target, -source),
            output_ndim=source.ndim - 1,
        )

    def _transpose_averages(self, target_cotangent: Array, /) -> Array:
        """Algebraic transpose of `apply` for fully active cells."""
        value = jnp.asarray(target_cotangent)
        trailing = (1,) * (value.ndim - 1)
        weights = (
            self.intersection_measures.astype(value.dtype)
            / self.target_volumes.astype(value.dtype)[self.target_routes]
        )
        weighted = value[self.target_routes] * weights.reshape((-1,) + trailing)
        source = jnp.zeros(
            (self.source_volumes.size,) + value.shape[1:], dtype=value.dtype
        )
        return source.at[self.source_indices].add(weighted)

    @checked
    def epoch_transition(
        self,
        source_field: DiscreteFieldSpace,
        target_field: DiscreteFieldSpace,
        source_epoch: TopologyEpoch,
        target_epoch: TopologyEpoch,
        /,
    ) -> TopologyEpochTransition:
        """Bind a fully covered cell remap as a topology-epoch transition.

        The epochs must realize both topology and geometry identities.
        Positivity and conservative content have the certified source-piece
        bounds; constant preservation is claimed only for geometric-overlap
        routes, not an area-changing chart deformation. Transpose uses CSR routing
        and Hilbert adjoint uses the FV physical-area pairings.
        """

        if (
            source_epoch.topology_id != self.source_topology_id
            or target_epoch.topology_id != self.target_topology_id
            or source_epoch.geometry_id != self.source_geometry_id
            or target_epoch.geometry_id != self.target_geometry_id
        ):
            raise ValueError(
                "remap.epochs must realize the remap topology and geometry IDs; got incompatible endpoints."
            )
        if not self.require_complete:
            raise ValueError(
                "remap.require_complete must be true for an epoch transition; got false."
            )
        primal = FunctionLinearOperator(
            lambda values: self.apply(values),
            source=source_field.vector_space,
            target=target_field.vector_space,
            transpose_action=self._transpose_averages,
            operator_id=f"{self.plan_id}:cell-averages",
        )
        exact_restriction = (
            self.method == "mapped-nested" and self.surface_chart_deformation is None
        )
        geometry_defect = 0.0
        if not exact_restriction:
            covered = np.bincount(
                np.asarray(self.target_routes),
                weights=np.asarray(self.intersection_measures),
                minlength=self.target_volumes.size,
            )
            uncertainty = np.bincount(
                np.asarray(self.target_routes),
                weights=np.asarray(self.intersection_error_bounds),
                minlength=self.target_volumes.size,
            ) + np.asarray(self.target_volume_error_bounds)
            lower_measure = np.asarray(self.target_volumes) - np.asarray(
                self.target_volume_error_bounds
            )
            geometry_defect = float(
                np.nextafter(
                    np.max(
                        (np.abs(covered - np.asarray(self.target_volumes)) + uncertainty)
                        / lower_measure,
                    ),
                    np.inf,
                )
            )
        transfer = FieldTransfer(
            source_field,
            target_field,
            primal,
            dual_pullback_operator=transpose(primal),
            hilbert_adjoint_operator=adjoint(primal),
            geometry=TransferGeometryBinding(
                self.source_geometry_id,
                self.target_geometry_id,
                "exact-restriction" if exact_restriction else "bounded-reconstruction",
                source_topology_id=self.source_topology_id,
                target_topology_id=self.target_topology_id,
                coverage_defect=geometry_defect,
            ),
            properties=TransferProperties(
                constant_preserving=self.surface_chart_deformation is None,
                conservative=True,
                positivity_preserving=True,
                adjoint_paired=True,
                differentiable_geometry=False,
                exact_on=("constants",) if self.surface_chart_deformation is None else (),
            ),
        )
        source_volumes = np.asarray(self.source_volumes, dtype=np.float64)
        target_volumes = np.asarray(self.target_volumes, dtype=np.float64)
        components = primal.source.size // source_volumes.size
        coverage = coverage_ledger(
            np.asarray(self.target_routes),
            np.asarray(self.source_indices),
            np.asarray(self.intersection_measures, dtype=np.float64),
            source_volumes,
            target_volumes,
            float(np.asarray(self.report.tolerance)),
            intersection_error_bounds=np.asarray(self.intersection_error_bounds),
            source_error_bounds=np.asarray(self.source_volume_error_bounds),
            target_error_bounds=np.asarray(self.target_volume_error_bounds),
        )
        source_limit = coverage.source_limit
        if self.surface_chart_contents is not None:
            source_limit = np.asarray(
                self.surface_chart_contents.source_content_defect_bounds, dtype=np.float64
            )
        return TopologyEpochTransition(
            source_epoch,
            target_epoch,
            transfer,
            np.repeat(source_volumes, components),
            np.repeat(target_volumes, primal.target.size // target_volumes.size),
            measure_defect_bound=np.repeat(source_limit, components),
        )


class UnstructuredRemapLimiter(StrEnum):
    """Slope limiter of the second-order unstructured remap."""

    NONE = "none"
    BARTH_JESPERSEN = "barth-jespersen"


class UnstructuredSecondOrderRemapResult(StrictModule):
    """Target averages of one second-order remap with limiter and ledger evidence.

    ``limiter_factors`` holds the slope factor of every source cell and payload
    component; ``limited_count`` counts the factors below one.
    ``conservation_residual_before`` is source content minus target content of
    the limited reconstruction.  That residual is redistributed into target cells
    in proportion to their volume (no limiter) or to their slack against the
    neighborhood bounds (Barth-Jespersen), and ``conservation_residual_after`` is
    the measured residual of the returned values.  ``restored`` is false where
    the bounded slack could not absorb the whole residual.
    """

    values: Array
    limiter_factors: Array
    limited_count: Array
    limited_fraction: Array
    minimum_limiter_factor: Array
    conservation_residual_before: Array
    conservation_residual_after: Array
    redistributed_content: Array
    restored: Array


def _vertex_neighbor_stencils(mesh: CellMesh, /) -> tuple[np.ndarray, np.ndarray]:
    """Ascending distinct cells sharing a vertex with each cell, padded by row.

    Vertex stars reach across corners, so boundary and corner cells keep a
    full-rank gradient stencil on meshes where face neighbors do not.
    """
    counts = np.asarray(tuple(block.cell_count for block in mesh.blocks), np.int64)
    starts = np.cumsum(counts) - counts
    cell_count = np.sum(counts)
    cells = np.concatenate(
        tuple(
            np.broadcast_to(
                np.arange(start, start + block.cell_count, dtype=np.int64)[:, None],
                block.vertices.shape,
            )[np.asarray(block.vertex_valid)]
            for start, block in zip(starts, mesh.blocks, strict=True)
        )
    )
    vertices = np.concatenate(
        tuple(
            np.asarray(block.vertices, dtype=np.int64)[np.asarray(block.vertex_valid)]
            for block in mesh.blocks
        )
    )
    star_cells = cells[np.lexsort((cells, vertices))]
    star_counts = np.bincount(vertices, minlength=mesh.coordinates.shape[0])
    star_starts = np.cumsum(star_counts) - star_counts
    degree = star_counts[vertices]
    pair_rows = np.repeat(cells, degree)
    local = np.arange(pair_rows.size) - np.repeat(np.cumsum(degree) - degree, degree)
    pair_columns = star_cells[np.repeat(star_starts[vertices], degree) + local]
    distinct = pair_rows != pair_columns
    keys = np.unique(pair_rows[distinct] * cell_count + pair_columns[distinct])
    rows = keys // cell_count
    row_counts = np.bincount(rows, minlength=cell_count)
    slots = np.arange(keys.size) - np.repeat(
        np.cumsum(row_counts) - row_counts, row_counts
    )
    width = np.max(row_counts, initial=1)
    neighbors = np.zeros((cell_count, width), dtype=np.int32)
    valid = np.zeros((cell_count, width), dtype=np.bool_)
    neighbors[rows, slots] = keys % cell_count
    valid[rows, slots] = True
    return neighbors, valid


def _least_squares_gradient(
    centroids: np.ndarray,
    measures: np.ndarray,
    neighbors: np.ndarray,
    valid: np.ndarray,
    maximum_condition: float,
    /,
) -> SmallLinearSolveResult:
    """Weights of ``u_n - u_s`` in each cell's weighted least-squares gradient.

    Solves the scaled normal equations
    ``sum_n w_n d_n d_n^T g = sum_n w_n d_n (u_n - u_s)`` with
    ``d_n = (c_n - c_s) / h_s``, ``h_s = |s|^(1/d)``, and inverse-square-distance
    weights ``w_n = |d_n|^-2`` for all cells in one batched small solve whose
    rank and condition evidence is returned.
    """
    dimension = centroids.shape[1]
    lengths = measures ** (1.0 / dimension)
    offsets = (centroids[neighbors] - centroids[:, None, :]) / lengths[:, None, None]
    distance_squared = np.sum(offsets * offsets, axis=-1)
    if np.any(valid & ~(distance_squared > 0.0)):
        raise ValueError(
            "remap.gradient.stencil must contain distinct cell centroids; got a zero centroid separation."
        )
    weights = np.where(valid, 1.0 / np.where(valid, distance_squared, 1.0), 0.0)
    normal = ein.contract("cw,cwi,cwj->cij", weights, offsets, offsets)
    right = ein.contract("cw,cwi->ciw", weights, offsets) / lengths[:, None, None]
    return solve_small_linear(
        SmallLinearSolvePlan(dimension, maximum_condition=maximum_condition),
        normal,
        right,
    )


def _barth_jespersen_factors(
    overlap_sources: EdgeRelation,
    entry_values: Array,
    increments: Array,
    entry_lower: Array,
    entry_upper: Array,
    /,
) -> Array:
    """Largest slope factor per source keeping every overlap average in bounds."""
    headroom = jnp.where(
        increments > 0.0, entry_upper - entry_values, entry_lower - entry_values
    )
    active = increments != 0.0
    ratio = jnp.where(active, headroom / jnp.where(active, increments, 1.0), 1.0)
    return route_reduce(overlap_sources, jnp.clip(ratio, 0.0, 1.0), reduction="min")


def _content_residual(source_content: Array, target_content: Array, /) -> Array:
    return compensated_sum_chunks((source_content, -target_content), output_ndim=1)


class _SecondOrderGradient(StrictModule):
    centroids: np.ndarray
    columns: np.ndarray
    columns_valid: np.ndarray
    solution: SmallLinearSolveResult
    gradient: SparseLinearMap
    gradient_id: str = eqx.field(static=True)


def _prepare_second_order_gradient(
    refinement: supermesh_owner.PreparedCommonRefinement,
    source: UnstructuredFiniteVolumeDiscretization,
    condition_limit: float,
) -> _SecondOrderGradient:
    measures = np.asarray(refinement.source_measures, dtype=np.float64)
    centroids = (
        np.asarray(refinement.source_first_moments, dtype=np.float64) / measures[:, None]
    )
    neighbors, valid = _vertex_neighbor_stencils(source.mesh)
    solution = _least_squares_gradient(
        centroids, measures, neighbors, valid, condition_limit
    )
    successful = np.asarray(solution.successful)
    if not np.all(successful):
        failed = np.flatnonzero(~successful)
        raise ValueError(
            f"remap.gradient must have full-rank stencils within maximum_condition; got rank deficient or ill-conditioned stencils for {failed.size} source cells (first: cell {failed[0]})."
        )
    count, dimension = centroids.shape
    width = neighbors.shape[1] + 1
    columns = np.concatenate(
        (neighbors, np.arange(count, dtype=np.int32)[:, None]), axis=1
    )
    columns_valid = np.concatenate((valid, np.ones((count, 1), dtype=np.bool_)), axis=1)
    # g_s = sum_n a_n (u_n - u_s): the owner column carries -sum_n a_n.
    neighbor_weights = np.where(valid[:, None, :], np.asarray(solution.value), 0.0)
    gradient_weights = np.concatenate(
        (neighbor_weights, -np.sum(neighbor_weights, axis=-1, keepdims=True)),
        axis=-1,
    )
    gradient_id = canonical_fingerprint(
        {
            "kind": "unstructured-remap-least-squares-gradient",
            "source": source.prepared_id,
            "centroids": array_tree_fingerprint(centroids),
            "stencil": array_tree_fingerprint(columns),
            "stencil_valid": array_tree_fingerprint(columns_valid),
            "weights": array_tree_fingerprint(gradient_weights),
        }
    )
    gradient = SparseLinearMap(
        RowRelation(
            np.broadcast_to(columns[:, None, :], (count, dimension, width)),
            source_size=count,
            valid=np.broadcast_to(columns_valid[:, None, :], (count, dimension, width)),
        ),
        gradient_weights,
        operator_id=gradient_id,
    )
    return _SecondOrderGradient(
        centroids, columns, columns_valid, solution, gradient, gradient_id
    )


class UnstructuredSecondOrderRemapPlan(StrictModule, NonTrainableState):
    """Bound-preserving second-order conservative remap on a common refinement.

    Source cell ``s`` carries the linear reconstruction
    ``u_s + phi_s g_s . (x - c_s)`` about its certified decomposition centroid
    ``c_s``, where ``g_s = G u`` is the weighted least-squares gradient over the
    cells sharing a vertex with ``s`` (a prepared sparse map). Its integral
    over overlap ``e = (s, t)`` with measure ``V_e`` and first moment ``m_e`` is
    ``I_e = V_e u_s + phi_s g_s . (m_e - c_s V_e)``, so linear fields are remapped
    exactly and every source cell keeps its content for any ``phi_s``.

    ``BARTH_JESPERSEN`` chooses the largest ``phi_s`` in ``[0, 1]`` that keeps
    every overlap average of ``s`` within the extrema of ``s`` and its stencil,
    so every target average is a convex combination of bounded overlap averages.
    ``apply`` is pure JAX in the field values and in the stored overlap and
    centroid geometry for the frozen combinatorics.
    """

    remap: UnstructuredConservativeRemapPlan
    overlap_first_moments: Array
    source_centroids: Array
    gradient: SparseLinearMap
    neighborhood: RowRelation
    overlap: EdgeRelation
    overlap_sources: EdgeRelation
    gradient_rank: Array
    gradient_condition: Array
    limiter: UnstructuredRemapLimiter = eqx.field(static=True)
    refinement_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        remap: UnstructuredConservativeRemapPlan,
        refinement: supermesh_owner.PreparedCommonRefinement,
        source: UnstructuredFiniteVolumeDiscretization,
        /,
        *,
        limiter: UnstructuredRemapLimiter = UnstructuredRemapLimiter.BARTH_JESPERSEN,
        maximum_condition: float = 1.0e12,
    ) -> None:

        limiter = parse(limiter, UnstructuredRemapLimiter, "remap.limiter")
        condition_limit = float(maximum_condition)
        if not np.isfinite(condition_limit) or condition_limit <= 1.0:
            raise ValueError(
                "remap.maximum_condition must be finite and greater than one; got an invalid limit."
            )
        if not refinement.succeeded:
            raise ValueError(
                f"remap.second_order.refinement must have SUCCESS status; got {refinement.status.name}: {refinement.evidence.reason}."
            )
        if not remap.require_complete:
            raise ValueError(
                "remap.second_order.require_complete must be true; got an incomplete-coverage plan."
            )
        if (
            source.topology_id != remap.source_topology_id
            or source.geometry_id != remap.source_geometry_id
        ):
            raise ValueError(
                "remap.second_order.source must bind the remap source geometry; got mismatched IDs."
            )
        if (
            refinement.dimension != source.cell_dimension
            or refinement.source_cell_count != source.cell_count
            or refinement.target_cell_count != remap.target_volumes.size
            or not np.array_equal(
                np.asarray(refinement.target_offsets), np.asarray(remap.target_offsets)
            )
            or not np.array_equal(
                np.asarray(refinement.source_cells), np.asarray(remap.source_indices)
            )
            or not np.array_equal(
                np.asarray(refinement.volumes),
                np.asarray(remap.intersection_measures),
            )
        ):
            raise ValueError(
                "remap.second_order.refinement must realize the remap overlaps; got incompatible counts, routes, or measures."
            )
        prepared_gradient = _prepare_second_order_gradient(
            refinement, source, condition_limit
        )
        centroids = prepared_gradient.centroids
        columns = prepared_gradient.columns
        columns_valid = prepared_gradient.columns_valid
        solution = prepared_gradient.solution
        gradient = prepared_gradient.gradient
        gradient_id = prepared_gradient.gradient_id
        count = centroids.shape[0]
        overlap = EdgeRelation(
            remap.source_indices,
            remap.target_routes,
            source_size=count,
            target_size=remap.target_volumes.size,
        )
        self.remap = remap
        self.overlap_first_moments = jnp.asarray(
            refinement.first_moments, dtype=jnp.float64
        )
        self.source_centroids = jnp.asarray(centroids)
        self.gradient = gradient
        self.neighborhood = RowRelation(columns, source_size=count, valid=columns_valid)
        self.overlap = overlap
        self.overlap_sources = overlap.transpose()
        self.gradient_rank = solution.rank
        self.gradient_condition = solution.condition_estimate
        self.limiter = limiter
        self.refinement_id = refinement.refinement_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "unstructured-second-order-remap",
                "remap": remap.plan_id,
                "refinement": refinement.refinement_id,
                "source": source.prepared_id,
                "limiter": limiter.value,
                "maximum_condition": condition_limit,
                "gradient": gradient_id,
            }
        )

    def apply(
        self, source_cell_averages: ArrayLike, /
    ) -> UnstructuredSecondOrderRemapResult:
        """Remap cell averages through the limited linear reconstruction."""
        remap = self.remap
        value = jnp.asarray(source_cell_averages)
        if value.ndim == 0 or value.shape[0] != remap.source_volumes.size:
            raise ValueError(
                "remap.second_order.values must begin with the source cell count; got a different shape."
            )
        if not jnp.issubdtype(value.dtype, jnp.floating):
            raise TypeError(
                "remap.second_order.values must have real floating dtype; got another dtype."
            )
        payload_shape = value.shape[1:]
        source = value.reshape((value.shape[0], -1))
        dtype = source.dtype
        measures = remap.intersection_measures.astype(dtype)[:, None]
        target_volumes = remap.target_volumes.astype(dtype)[:, None]
        source_content = remap.source_volumes.astype(dtype)[:, None] * source
        centroid_offsets = self.overlap_first_moments.astype(
            dtype
        ) / measures - gather_routes(self.overlap, self.source_centroids.astype(dtype))
        increments = ein.contract(
            "ed,edc->ec",
            centroid_offsets,
            gather_routes(self.overlap, self.gradient.mv(source)),
        )
        entry_values = gather_routes(self.overlap, source)
        match self.limiter:
            case UnstructuredRemapLimiter.NONE:
                factors = jnp.ones_like(source)
                target_content = route_reduce(
                    self.overlap, measures * (entry_values + increments)
                )
                residual = _content_residual(source_content, target_content)
                capacity = jnp.broadcast_to(target_volumes, target_content.shape)
                amount = residual
            case UnstructuredRemapLimiter.BARTH_JESPERSEN:
                stencil_values = gather_routes(self.neighborhood, source)
                lower = route_reduce(self.neighborhood, stencil_values, reduction="min")
                upper = route_reduce(self.neighborhood, stencil_values, reduction="max")
                entry_lower = gather_routes(self.overlap, lower)
                entry_upper = gather_routes(self.overlap, upper)
                factors = _barth_jespersen_factors(
                    self.overlap_sources,
                    entry_values,
                    increments,
                    entry_lower,
                    entry_upper,
                )
                target_content = route_reduce(
                    self.overlap,
                    measures
                    * (entry_values + gather_routes(self.overlap, factors) * increments),
                )
                residual = _content_residual(source_content, target_content)
                target_lower = route_reduce(self.overlap, entry_lower, reduction="min")
                target_upper = route_reduce(self.overlap, entry_upper, reduction="max")
                capacity = jnp.maximum(
                    jnp.where(
                        residual > 0.0,
                        target_volumes * target_upper - target_content,
                        target_content - target_volumes * target_lower,
                    ),
                    0.0,
                )
                amount = jnp.sign(residual) * jnp.minimum(
                    jnp.abs(residual), jnp.sum(capacity, axis=0)
                )
            case _:
                raise ValueError(
                    f"remap.limiter must be a supported selector; got {self.limiter!r}."
                )
        available = jnp.sum(capacity, axis=0)
        restored_content = (
            target_content
            + capacity / jnp.where(available > 0.0, available, 1.0) * amount
        )
        limited_count = jnp.sum(factors < 1.0, dtype=jnp.int32)
        return UnstructuredSecondOrderRemapResult(
            values=(restored_content / target_volumes).reshape(
                (remap.target_volumes.size,) + payload_shape
            ),
            limiter_factors=factors.reshape((source.shape[0],) + payload_shape),
            limited_count=limited_count,
            limited_fraction=limited_count.astype(dtype) / factors.size,
            minimum_limiter_factor=jnp.min(factors),
            conservation_residual_before=residual.reshape(payload_shape),
            conservation_residual_after=_content_residual(
                source_content, restored_content
            ).reshape(payload_shape),
            redistributed_content=amount.reshape(payload_shape),
            restored=(amount == residual).reshape(payload_shape),
        )


__all__ = [
    "UnstructuredConservativeRemapPlan",
    "UnstructuredRemapLimiter",
    "UnstructuredRemapReport",
    "UnstructuredSecondOrderRemapPlan",
    "RemapCoverageLedger",
    "UnstructuredSecondOrderRemapResult",
]
