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


def _boundary_support_id(atlas: BoundaryAtlas, /) -> str:
    return canonical_fingerprint(
        {
            "kind": "boundary-singular-support-v1",
            "source_id": atlas.source_id,
            "entity_ids": array_tree_fingerprint(atlas.source_entity_ids),
            "orientation": array_tree_fingerprint(atlas.orientation),
            "mapping": repr(atlas.mapping),
            "trim_domains": repr(atlas.trim_domains),
        }
    )


class BoundaryPanelization2D(StrictModule, NonTrainableState):
    """Fixed Gauss-Legendre panelization of a two-dimensional boundary atlas."""

    atlas: BoundaryAtlas
    geometry: CompiledGeometry | None
    chart_indices: Array
    references: Array
    points: Array
    normals: Array
    weights: Array
    panel_ids: Array
    panels_per_chart: int = eqx.field(static=True)
    quadrature_order: int = eqx.field(static=True)
    source_support_id: str = eqx.field(static=True)
    panelization_id: str = eqx.field(static=True)

    def __init__(
        self,
        atlas: BoundaryAtlas,
        /,
        *,
        panels_per_chart: int,
        quadrature_order: int,
        geometry: CompiledGeometry | None = None,
    ):
        if not isinstance(atlas, BoundaryAtlas):
            raise TypeError("BoundaryPanelization2D requires a BoundaryAtlas.")
        if atlas.ambient_dimension != 2 or atlas.reference_dimension != 1:
            raise ValueError("BoundaryPanelization2D requires curve charts in 2D.")
        panels = int(panels_per_chart)
        order = int(quadrature_order)
        if panels <= 0 or order < 2:
            raise ValueError("Panel count must be positive and quadrature order at least two.")
        if geometry is not None:
            if not isinstance(geometry, CompiledGeometry):
                raise TypeError("Panelization geometry must be a CompiledGeometry.")
            if not geometry.has_capability(GeometryCapability.BOUNDARY_ATLAS):
                raise TypeError(
                    "Panelization geometry must provide a boundary atlas."
                )
            if geometry.ambient_dimension != 2:
                raise ValueError("Panelization geometry must be two-dimensional.")
            if _boundary_support_id(geometry.boundary_atlas) != _boundary_support_id(atlas):
                raise ValueError(
                    "Panelization geometry and boundary atlas must describe "
                    "the same support."
                )
        nodes, weights = np.polynomial.legendre.leggauss(order)
        references = []
        chart_indices = []
        panel_ids = []
        reference_weights = []
        panel_id = 0
        for chart in range(atlas.num_charts):
            if not bool(atlas.seam_owner[chart]):
                continue
            for panel in range(panels):
                lower = panel / panels
                upper = (panel + 1) / panels
                mapped = lower + 0.5 * (nodes + 1.0) * (upper - lower)
                references.extend(mapped[:, None])
                chart_indices.extend([chart] * order)
                panel_ids.extend([panel_id] * order)
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
        self.chart_indices = chart_array
        self.references = reference_array
        self.points = frame.origin
        self.normals = frame.normal
        self.weights = physical_weights
        self.panel_ids = jnp.asarray(panel_ids, dtype=jnp.int32)
        self.panels_per_chart = panels
        self.quadrature_order = order
        self.source_support_id = source_support_id
        self.panelization_id = canonical_fingerprint(
            {
                "kind": "boundary-panelization-2d-v1",
                "source_support_id": source_support_id,
                "panels_per_chart": panels,
                "quadrature_order": order,
                "chart_indices": array_tree_fingerprint(chart_array),
                "references": array_tree_fingerprint(reference_array),
            }
        )

    @property
    def node_count(self) -> int:
        return int(self.points.shape[0])

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
        if any(not geometry.has_capability(capability) for capability in required_capabilities):
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
        classification_tolerance = (
            64.0 * jnp.finfo(values.dtype).eps * scale
        )
        on_boundary = jnp.abs(signed_distance) <= classification_tolerance
        if target_side == "interior":
            side_matches = jnp.all(
                inside & (signed_distance < -classification_tolerance)
            )
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
                "boundary_classification_tolerance": float(
                    classification_tolerance
                ),
            }
        )


class BoundaryLayerApproximationReport(StrictModule, NonTrainableState):
    """Quadrature and density approximation evidence, separate from PDE exactness."""
    panelization_id: str = eqx.field(static=True)
    kernel_id: str = eqx.field(static=True)
    density_space: str = eqx.field(static=True)
    trace_policy: str = eqx.field(static=True)
    quadrature_order: int = eqx.field(static=True)
    panel_count: int = eqx.field(static=True)
    node_count: int = eqx.field(static=True)
    approximation_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        panelization: BoundaryPanelization2D,
        kernel_id: str,
        density_space: str,
        trace_policy: str,
    ):
        if not kernel_id or not density_space or not trace_policy:
            raise ValueError("Boundary-layer approximation identifiers must be nonempty.")
        self.panelization_id = panelization.panelization_id
        self.kernel_id = str(kernel_id)
        self.density_space = str(density_space)
        self.trace_policy = str(trace_policy)
        self.quadrature_order = panelization.quadrature_order
        self.panel_count = int(jnp.max(panelization.panel_ids)) + 1
        self.node_count = panelization.node_count
        self.approximation_id = canonical_fingerprint(
            {
                "kind": "boundary-layer-approximation-v1",
                "panelization_id": self.panelization_id,
                "kernel_id": self.kernel_id,
                "density_space": self.density_space,
                "trace_policy": self.trace_policy,
            }
        )


__all__ = [
    "AbstractLayerKernel",
    "BoundaryLayerApproximationReport",
    "BoundaryPanelization2D",
    "KernelActionSide",
    "LayerPotentialTargetReport",
]
