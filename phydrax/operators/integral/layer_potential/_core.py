#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
import math
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....equations.trefftz._core import (
    AbstractTrialSpaceAdmissibility,
    trial_target_fingerprint,
)
from ....geometry import (
    BoundaryAtlas,
    CompiledGeometry,
    GeometryCapability,
    SignReliability,
    ZeroSetAccuracy,
)


KernelActionSide = Literal["left", "right"]


class AbstractLayerKernel(StrictModule, NonTrainableState):
    """Mathematical layer kernel independent of geometry discretization."""

    @property
    @abc.abstractmethod
    def ambient_dimension(self) -> int:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def source_event_shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def target_event_shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def action_side(self) -> KernelActionSide:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def kernel_id(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def value(self, target: Array, source: Array, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def source_normal_derivative(
        self,
        target: Array,
        source: Array,
        source_normal: Array,
        /,
    ) -> Array:
        raise NotImplementedError


def _type_identity(value: object, /) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _boundary_support_id(atlas: BoundaryAtlas, /) -> str:
    return canonical_fingerprint(
        {
            "kind": "boundary-singular-support-v2",
            "source_id": atlas.source_id,
            "atlas_type": _type_identity(atlas),
            "mapping_type": _type_identity(atlas.mapping),
            "trim_types": [
                None if trim is None else _type_identity(trim)
                for trim in atlas.trim_domains
            ],
            "atlas_arrays": array_tree_fingerprint(atlas),
            "mapping_arrays": array_tree_fingerprint(atlas.mapping),
            "trim_arrays": array_tree_fingerprint(atlas.trim_domains),
            "mapping_structure": repr(atlas.mapping),
            "trim_structure": repr(atlas.trim_domains),
        }
    )


class BoundaryCornerTopology2D(StrictModule, NonTrainableState):
    """Declared chart endpoints and opening angles for boundary corners."""

    corner_chart_ends: tuple[tuple[int, Literal["start", "end"]], ...] = eqx.field(
        static=True
    )
    interior_angles: Array
    topology_id: str = eqx.field(static=True)

    def __init__(
        self,
        chart_count: int,
        corner_chart_ends: tuple[tuple[int, Literal["start", "end"]], ...],
        /,
        *,
        interior_angles: ArrayLike | None = None,
    ):
        count = int(chart_count)
        if count <= 0:
            raise ValueError("chart_count must be positive.")
        corners = tuple((int(chart), side) for chart, side in corner_chart_ends)
        if len(set(corners)) != len(corners):
            raise ValueError("Corner chart endpoints must be unique.")
        if any(
            chart < 0 or chart >= count or side not in ("start", "end")
            for chart, side in corners
        ):
            raise ValueError("Corner chart endpoints must reference valid charts.")
        angles = (
            jnp.full((len(corners),), jnp.nan)
            if interior_angles is None
            else jnp.asarray(interior_angles, dtype=float).reshape((-1,))
        )
        if angles.shape != (len(corners),):
            raise ValueError("interior_angles must match corner_chart_ends.")
        if bool(jnp.any(jnp.isfinite(angles) & (angles <= 0.0))):
            raise ValueError("Finite corner angles must be positive.")
        self.corner_chart_ends = corners
        self.interior_angles = angles
        self.topology_id = canonical_fingerprint(
            {
                "kind": "boundary-corner-topology-2d-v1",
                "chart_count": count,
                "corner_chart_ends": corners,
                "interior_angles": array_tree_fingerprint(angles),
            }
        )


class BoundaryPanelPartition2D(StrictModule, NonTrainableState):
    """Validated reference breakpoints with optional endpoint grading."""

    breakpoints: tuple[tuple[float, ...], ...] = eqx.field(static=True)
    panels_per_chart: int = eqx.field(static=True)
    grading: Literal["uniform", "kress", "dyadic"] = eqx.field(static=True)
    grading_order: int = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    partition_id: str = eqx.field(static=True)

    def __init__(
        self,
        atlas: BoundaryAtlas,
        panels_per_chart: int,
        /,
        *,
        grading: Literal["uniform", "kress", "dyadic"] = "uniform",
        grading_order: int = 3,
        corner_topology: BoundaryCornerTopology2D | None = None,
    ):
        if not isinstance(atlas, BoundaryAtlas):
            raise TypeError("BoundaryPanelPartition2D requires a BoundaryAtlas.")
        panels = int(panels_per_chart)
        order = int(grading_order)
        if panels <= 0:
            raise ValueError("panels_per_chart must be positive.")
        if grading not in ("uniform", "kress", "dyadic"):
            raise ValueError("Unknown boundary panel grading.")
        if order < 2:
            raise ValueError("grading_order must be at least two.")
        topology = (
            BoundaryCornerTopology2D(atlas.num_charts, ())
            if corner_topology is None
            else corner_topology
        )
        if not isinstance(topology, BoundaryCornerTopology2D):
            raise TypeError("corner_topology must be BoundaryCornerTopology2D.")
        starts = {chart for chart, side in topology.corner_chart_ends if side == "start"}
        ends = {chart for chart, side in topology.corner_chart_ends if side == "end"}

        def grade(value: float, start: bool, end: bool) -> float:
            if grading == "uniform" or not (start or end):
                return value
            if grading == "dyadic":
                if start and end:
                    left = value**order
                    right = 1.0 - (1.0 - value) ** order
                    return 0.5 * (left + right)
                if start:
                    return value**order
                return 1.0 - (1.0 - value) ** order
            power = float(order)
            numerator = value**power
            denominator = numerator + (1.0 - value) ** power
            if start and end:
                return numerator / denominator if 0.0 < value < 1.0 else value
            if start:
                return value**power
            return 1.0 - (1.0 - value) ** power

        breaks = []
        for chart in range(atlas.num_charts):
            coordinates = tuple(
                grade(index / panels, chart in starts, chart in ends)
                for index in range(panels + 1)
            )
            if any(right <= left for left, right in zip(coordinates[:-1], coordinates[1:], strict=True)):
                raise ValueError("Grading produced non-increasing panel breakpoints.")
            breaks.append(coordinates)
        self.breakpoints = tuple(breaks)
        self.panels_per_chart = panels
        self.grading = grading
        self.grading_order = order
        self.topology_id = topology.topology_id
        self.partition_id = canonical_fingerprint(
            {
                "kind": "boundary-panel-partition-2d-v1",
                "source_support_id": _boundary_support_id(atlas),
                "breakpoints": breaks,
                "grading": grading,
                "grading_order": order,
                "topology_id": topology.topology_id,
            }
        )


class BoundaryPanelization2D(StrictModule, NonTrainableState):
    """Fixed Gauss-Legendre panelization of a two-dimensional boundary atlas."""

    atlas: BoundaryAtlas
    geometry: CompiledGeometry | None
    partition: BoundaryPanelPartition2D
    chart_indices: Array
    references: Array
    points: Array
    normals: Array
    weights: Array
    panel_ids: Array
    panel_chart_indices: Array
    panel_reference_bounds: Array
    panels_per_chart: int = eqx.field(static=True)
    quadrature_order: int = eqx.field(static=True)
    source_support_id: str = eqx.field(static=True)
    panelization_id: str = eqx.field(static=True)

    def __init__(
        self,
        atlas: BoundaryAtlas,
        /,
        *,
        panels_per_chart: int | None = None,
        quadrature_order: int,
        geometry: CompiledGeometry | None = None,
        partition: BoundaryPanelPartition2D | None = None,
    ):
        if not isinstance(atlas, BoundaryAtlas):
            raise TypeError("BoundaryPanelization2D requires a BoundaryAtlas.")
        if atlas.ambient_dimension != 2 or atlas.reference_dimension != 1:
            raise ValueError("BoundaryPanelization2D requires curve charts in 2D.")
        if partition is None:
            if panels_per_chart is None:
                raise ValueError(
                    "Specify panels_per_chart or a BoundaryPanelPartition2D."
                )
            partition_ = BoundaryPanelPartition2D(atlas, panels_per_chart)
        else:
            if not isinstance(partition, BoundaryPanelPartition2D):
                raise TypeError(
                    "partition must be a BoundaryPanelPartition2D."
                )
            if len(partition.breakpoints) != atlas.num_charts:
                raise ValueError("Panel partition chart count does not match the atlas.")
            if (
                panels_per_chart is not None
                and int(panels_per_chart) != partition.panels_per_chart
            ):
                raise ValueError(
                    "panels_per_chart conflicts with the supplied partition."
                )
            partition_ = partition
        panels = partition_.panels_per_chart
        order = int(quadrature_order)
        if order < 2:
            raise ValueError("Quadrature order must be at least two.")
        if geometry is not None:
            if not isinstance(geometry, CompiledGeometry):
                raise TypeError("Panelization geometry must be a CompiledGeometry.")
            if not geometry.has_capability(GeometryCapability.BOUNDARY_ATLAS):
                raise TypeError("Panelization geometry must provide a boundary atlas.")
            if geometry.ambient_dimension != 2:
                raise ValueError("Panelization geometry must be two-dimensional.")
            if _boundary_support_id(geometry.boundary_atlas) != _boundary_support_id(
                atlas
            ):
                raise ValueError(
                    "Panelization geometry and boundary atlas must describe "
                    "the same support."
                )
        nodes, weights = np.polynomial.legendre.leggauss(order)
        references = []
        chart_indices = []
        panel_ids = []
        panel_chart_indices = []
        panel_reference_bounds = []
        reference_weights = []
        panel_id = 0
        for chart in range(atlas.num_charts):
            if not bool(atlas.seam_owner[chart]):
                continue
            chart_breakpoints = partition_.breakpoints[chart]
            for panel, (lower, upper) in enumerate(
                zip(chart_breakpoints[:-1], chart_breakpoints[1:], strict=True)
            ):
                mapped = lower + 0.5 * (nodes + 1.0) * (upper - lower)
                references.extend(mapped[:, None])
                chart_indices.extend([chart] * order)
                panel_ids.extend([panel_id] * order)
                panel_chart_indices.append(chart)
                panel_reference_bounds.append((lower, upper))
                reference_weights.extend(0.5 * (upper - lower) * weights)
                panel_id += 1
        if not references:
            raise ValueError("Boundary panelization has no owned charts.")
        chart_array = jnp.asarray(chart_indices, dtype=jnp.int32)
        reference_array = jnp.asarray(np.asarray(references), dtype=float)
        frame = atlas.frame(chart_array, reference_array)
        physical_weights = jnp.asarray(reference_weights) * frame.jacobian
        if not bool(jnp.all(jnp.isfinite(frame.origin))) or not bool(
            jnp.all(jnp.isfinite(physical_weights))
        ):
            raise ValueError("Boundary panelization geometry must be finite.")
        if bool(jnp.any(physical_weights <= 0.0)):
            raise ValueError("Boundary panelization weights must be positive.")
        source_support_id = _boundary_support_id(atlas)
        self.atlas = atlas
        self.geometry = geometry
        self.partition = partition_
        self.chart_indices = chart_array
        self.references = reference_array
        self.points = frame.origin
        self.normals = frame.normal
        self.weights = physical_weights
        self.panel_ids = jnp.asarray(panel_ids, dtype=jnp.int32)
        self.panel_chart_indices = jnp.asarray(
            panel_chart_indices, dtype=jnp.int32
        )
        self.panel_reference_bounds = jnp.asarray(
            panel_reference_bounds, dtype=float
        )
        self.panels_per_chart = panels
        self.quadrature_order = order
        self.source_support_id = source_support_id
        self.panelization_id = canonical_fingerprint(
            {
                "kind": "boundary-panelization-2d-v2",
                "source_support_id": source_support_id,
                "partition_id": partition_.partition_id,
                "quadrature_order": order,
                "chart_indices": array_tree_fingerprint(chart_array),
                "references": array_tree_fingerprint(reference_array),
            }
        )

    @property
    def node_count(self) -> int:
        return int(self.points.shape[0])
    @property
    def panel_count(self) -> int:
        return int(self.panel_chart_indices.shape[0])

    @property
    def boundary_measure(self) -> Array:
        return jnp.sum(self.weights)


class LayerPotentialTargetReport(AbstractTrialSpaceAdmissibility):
    """Certified continuous-boundary membership and numerical-accuracy evidence."""

    minimum_distance: Array
    boundary_classification_tolerance: Array
    intersects_singular_support: Array
    pde_membership_valid: Array
    requested_accuracy_clearance: Array
    accuracy_supported: Array
    target_count: int = eqx.field(static=True)
    singular_support_id: str = eqx.field(static=True)
    target_side: Literal["interior", "exterior", "boundary"] = eqx.field(static=True)
    target_fingerprint: str = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    def __init__(
        self,
        targets: ArrayLike,
        panelization: BoundaryPanelization2D,
        /,
        *,
        target_side: Literal["interior", "exterior", "boundary"],
        accuracy_clearance: float = 0.0,
    ):
        values = jnp.asarray(targets, dtype=float)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] == 0:
            raise ValueError("Layer targets must have shape (target_count, 2).")
        if target_side not in ("interior", "exterior", "boundary"):
            raise ValueError("Layer target side must be interior, exterior, or boundary.")
        geometry = panelization.geometry
        if geometry is None:
            raise TypeError(
                "Target admissibility requires the certified geometry used "
                "to build the panelization."
            )
        required_capabilities = (
            GeometryCapability.REGION_QUERY,
            GeometryCapability.SIGNED_DISTANCE,
        )
        if any(
            not geometry.has_capability(capability)
            for capability in required_capabilities
        ):
            raise TypeError(
                "Target admissibility requires certified region and signed-distance queries."
            )
        field_certificate = geometry.field_certificate
        if (
            field_certificate.zero_set_accuracy is not ZeroSetAccuracy.EXACT
            or field_certificate.sign_reliability is not SignReliability.RELIABLE
            or not field_certificate.is_signed_distance
        ):
            raise TypeError(
                "Target admissibility requires an exact, sign-reliable "
                "signed-distance certificate."
            )
        clearance = float(accuracy_clearance)
        if not math.isfinite(clearance) or clearance < 0.0:
            raise ValueError("Accuracy clearance must be finite and nonnegative.")

        signed_distance = jnp.asarray(geometry.signed_distance(values))
        inside = jnp.asarray(geometry.contains(values), dtype=bool)
        scale = jnp.maximum(jnp.max(jnp.abs(values)), 1.0)
        classification_tolerance = 64.0 * jnp.finfo(values.dtype).eps * scale
        on_boundary = jnp.abs(signed_distance) <= classification_tolerance
        if target_side == "interior":
            side_matches = jnp.all(inside & (signed_distance < -classification_tolerance))
        elif target_side == "exterior":
            side_matches = jnp.all(
                (~inside) & (signed_distance > classification_tolerance)
            )
        else:
            side_matches = jnp.all(on_boundary)
        intersects = jnp.any(on_boundary)
        membership = (~intersects) & side_matches
        minimum = jnp.min(jnp.abs(signed_distance))

        self.minimum_distance = minimum
        self.boundary_classification_tolerance = classification_tolerance
        self.intersects_singular_support = intersects
        self.pde_membership_valid = membership
        self.requested_accuracy_clearance = jnp.asarray(clearance)
        self.accuracy_supported = membership & (minimum >= clearance)
        self.target_count = int(values.shape[0])
        self.singular_support_id = panelization.source_support_id
        self.target_side = target_side
        self.target_fingerprint = trial_target_fingerprint(values, 2)
        self.report_id = canonical_fingerprint(
            {
                "kind": "layer-potential-target-report-v2",
                "singular_support_id": panelization.source_support_id,
                "target_fingerprint": self.target_fingerprint,
                "target_side": target_side,
                "accuracy_clearance": clearance,
                "boundary_classification_tolerance": float(classification_tolerance),
            }
        )


class BoundaryOperatorAssemblyReport(StrictModule, NonTrainableState):
    """Explicit trace-operator assembly status and corrected-block evidence."""

    panelization_id: str = eqx.field(static=True)
    kernel_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    trace_policy: str = eqx.field(static=True)
    corrected_block_count: int = eqx.field(static=True)
    block_status: Array
    block_errors: Array
    block_evaluations: Array
    status: Array
    error_estimate: Array
    accuracy_supported: Array
    assembly_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        panelization: BoundaryPanelization2D,
        kernel_id: str,
        policy_id: str,
        trace_policy: str,
        block_status: ArrayLike,
        block_errors: ArrayLike,
        block_evaluations: ArrayLike,
    ):
        statuses = jnp.asarray(block_status, dtype=jnp.int32).reshape((-1,))
        errors = jnp.asarray(block_errors).reshape((-1,))
        evaluations = jnp.asarray(block_evaluations, dtype=jnp.int32).reshape((-1,))
        if statuses.shape != errors.shape or statuses.shape != evaluations.shape:
            raise ValueError("Operator block diagnostics must have equal shapes.")
        if not kernel_id or not policy_id or not trace_policy:
            raise ValueError("Operator assembly identifiers must be nonempty.")
        finite = jnp.all(jnp.isfinite(errors))
        status = jnp.max(statuses) if statuses.size else jnp.asarray(0, dtype=jnp.int32)
        error = jnp.max(errors) if errors.size else jnp.asarray(0.0)
        supported = finite & jnp.all(statuses == 0)
        self.panelization_id = panelization.panelization_id
        self.kernel_id = str(kernel_id)
        self.policy_id = str(policy_id)
        self.trace_policy = str(trace_policy)
        self.corrected_block_count = int(statuses.size)
        self.block_status = statuses
        self.block_errors = errors
        self.block_evaluations = evaluations
        self.status = status
        self.error_estimate = error
        self.accuracy_supported = supported
        self.assembly_id = canonical_fingerprint(
            {
                "kind": "boundary-operator-assembly-v1",
                "panelization_id": self.panelization_id,
                "kernel_id": self.kernel_id,
                "policy_id": self.policy_id,
                "trace_policy": self.trace_policy,
                "block_status": array_tree_fingerprint(statuses),
                "block_errors": array_tree_fingerprint(errors),
            }
        )


class LayerDiscretizationReport(StrictModule, NonTrainableState):
    """Panel and density discretization evidence, separate from evaluation."""

    panelization_id: str = eqx.field(static=True)
    kernel_id: str = eqx.field(static=True)
    density_space: str = eqx.field(static=True)
    trace_policy: str = eqx.field(static=True)
    quadrature_order: int = eqx.field(static=True)
    panel_count: int = eqx.field(static=True)
    node_count: int = eqx.field(static=True)
    discretization_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        panelization: BoundaryPanelization2D,
        kernel_id: str,
        density_space: str,
        trace_policy: str,
    ):
        if not kernel_id or not density_space or not trace_policy:
            raise ValueError("Layer discretization identifiers must be nonempty.")
        self.panelization_id = panelization.panelization_id
        self.kernel_id = str(kernel_id)
        self.density_space = str(density_space)
        self.trace_policy = str(trace_policy)
        self.quadrature_order = panelization.quadrature_order
        self.panel_count = int(jnp.max(panelization.panel_ids)) + 1
        self.node_count = panelization.node_count
        self.discretization_id = canonical_fingerprint(
            {
                "kind": "layer-discretization-v1",
                "panelization_id": self.panelization_id,
                "kernel_id": self.kernel_id,
                "density_space": self.density_space,
                "trace_policy": self.trace_policy,
            }
        )


__all__ = [
    "AbstractLayerKernel",
    "BoundaryOperatorAssemblyReport",
    "BoundaryCornerTopology2D",
    "BoundaryPanelPartition2D",
    "BoundaryPanelization2D",
    "KernelActionSide",
    "LayerDiscretizationReport",
    "LayerPotentialTargetReport",
]
