#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import (
    FiniteVolumePlan,
    NonuniformCellAxisSpec,
    TensorGridPlan,
    UniformCellAxisSpec,
)
from ._free_surface_ale import FreeSurfaceALEState, GraphSurfaceALEPlan
from ._free_surface_step import (
    FreeSurfaceALEContinuationState,
    OnePhaseFreeSurfaceALEPlan,
    PreparedOnePhaseFreeSurfaceALE,
)


ShorelineStatus = Literal[
    "continue", "rezone", "handoff-hydrostatic", "handoff-two-phase", "reject"
]


class RezoneEvidence(StrictModule):
    scalar_content_defect: dict[str, Array]
    momentum_defect: Array
    kinetic_energy_defect: Array
    old_quality: Array
    new_quality: Array
    projection_residual: Array
    finite: Array
    conservative: Array
    successful: Array
    event_id: str = eqx.field(static=True)


class FreeSurfaceRezoneResult(StrictModule):
    hydrodynamics: PreparedOnePhaseFreeSurfaceALE
    state: FreeSurfaceALEContinuationState
    evidence: RezoneEvidence


class ShorelineEvent(StrictModule):
    minimum_height: Array
    maximum_slope: Array
    status_code: Array
    finite: Array
    valid: Array
    status: ShorelineStatus = eqx.field(static=True)
    target_product: str = eqx.field(static=True)
    event_id: str = eqx.field(static=True)


class FreeSurfaceRezonePlan(StrictModule, NonTrainableState):
    """Topology-preserving vertical redistribution and conservative remap."""

    stretching_exponent: float = eqx.field(static=True)
    minimum_quality_improvement: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        stretching_exponent: float = 1.5,
        /,
        *,
        minimum_quality_improvement: float = 0.0,
    ):
        exponent = float(stretching_exponent)
        improvement = float(minimum_quality_improvement)
        if (
            not np.isfinite(exponent)
            or exponent <= 0.0
            or not np.isfinite(improvement)
            or improvement < 0.0
        ):
            raise ValueError("Invalid free-surface rezone policy.")
        self.stretching_exponent = exponent
        self.minimum_quality_improvement = improvement
        self.plan_id = canonical_fingerprint(
            {
                "kind": "free-surface-vertical-rezone",
                "stretching_exponent": exponent,
                "minimum_quality_improvement": improvement,
            }
        )

    def _new_reference(self, hydrodynamics: PreparedOnePhaseFreeSurfaceALE):
        old = hydrodynamics.reference
        axes = old.grid.structured_axes
        bounds = jnp.stack(
            (
                jnp.asarray([axis.bounds[0] for axis in axes]),
                jnp.asarray([axis.bounds[1] for axis in axes]),
            )
        )
        specs = []
        for axis_index, axis in enumerate(axes):
            physical_edges = jnp.concatenate(
                (
                    axis.bounds[:1],
                    axis.bounds[0] + jnp.cumsum(axis.interval_widths),
                )
            )
            normalized = (physical_edges - physical_edges[0]) / (
                physical_edges[-1] - physical_edges[0]
            )
            if axis_index == 2:
                normalized = (
                    jnp.linspace(0.0, 1.0, normalized.size) ** self.stretching_exponent
                )
            specs.append(
                UniformCellAxisSpec(normalized.size - 1, periodic=axis.periodic)
                if normalized.size < 5
                else NonuniformCellAxisSpec(normalized, periodic=axis.periodic)
            )
        grid = TensorGridPlan(tuple(specs), axis_names=old.grid.axis_names).prepare(
            bounds
        )
        return FiniteVolumePlan(
            grid,
            component_names=old.component_names,
        ).prepare()

    @staticmethod
    def _column_vertical_vertices(vertices: Array, /) -> Array:
        return 0.25 * (
            vertices[:-1, :-1] + vertices[1:, :-1] + vertices[:-1, 1:] + vertices[1:, 1:]
        )

    @staticmethod
    def _vertical_overlap_content(
        old_vertices: Array,
        new_vertices: Array,
        old_content: Array,
        old_volume: Array,
        new_horizontal_area: Array,
    ) -> Array:
        old_lower = old_vertices[..., :-1, 2]
        old_upper = old_vertices[..., 1:, 2]
        new_lower = new_vertices[..., :-1, 2]
        new_upper = new_vertices[..., 1:, 2]
        overlap = jnp.maximum(
            0.0,
            jnp.minimum(new_upper[..., :, None], old_upper[..., None, :])
            - jnp.maximum(new_lower[..., :, None], old_lower[..., None, :]),
        )
        old_concentration = jnp.where(old_volume > 0.0, old_content / old_volume, 0.0)
        return new_horizontal_area[..., None] * jnp.sum(
            overlap * old_concentration[..., None, :], axis=-1
        )

    @staticmethod
    def _interpolate_velocity(
        value: Array,
        old_sigma: Array,
        new_sigma: Array,
        old_edges: Array,
        new_edges: Array,
        /,
    ) -> Array:
        moved = jnp.moveaxis(value, -1, -1)
        flat = moved.reshape((-1, moved.shape[-1]))
        old_points = old_edges if moved.shape[-1] == old_edges.size else old_sigma
        new_points = new_edges if moved.shape[-1] == old_edges.size else new_sigma
        interpolated = jax.vmap(lambda row: jnp.interp(new_points, old_points, row))(flat)
        return interpolated.reshape(moved.shape[:-1] + (new_points.size,))

    def rezone(
        self,
        hydrodynamics: PreparedOnePhaseFreeSurfaceALE,
        continuation: FreeSurfaceALEContinuationState,
        /,
    ) -> FreeSurfaceRezoneResult:
        old_surface = hydrodynamics.surface
        old_state = continuation.state
        old_view = hydrodynamics.view(old_state, continuation.eta_rate)
        old_geometry = old_view.geometry
        new_reference = self._new_reference(hydrodynamics)
        new_surface_plan = GraphSurfaceALEPlan(
            new_reference,
            old_surface.plan.bottom,
            minimum_height=old_surface.plan.minimum_height,
            maximum_slope=old_surface.plan.maximum_slope,
            tolerance=old_surface.plan.tolerance,
            maximum_iterations=old_surface.plan.maximum_iterations,
        )
        new_plan = OnePhaseFreeSurfaceALEPlan(
            new_surface_plan,
            boundary=hydrodynamics.plan.boundary,
            density=hydrodynamics.plan.density,
            gravity=hydrodynamics.plan.gravity,
            surface_tension=hydrodynamics.plan.surface_tension,
            wave=hydrodynamics.plan.wave,
            coupling_iterations=hydrodynamics.plan.coupling_iterations,
            coupling_tolerance=hydrodynamics.plan.coupling_tolerance,
        )
        new_hydrodynamics = new_plan.prepare()
        new_geometry = new_hydrodynamics.surface.geometry(
            jnp.asarray(0.0),
            old_state.eta,
            continuation.eta_rate,
        )
        old_points = old_surface.plan.reference.grid.structured_axes[2].point_coordinates
        new_points = new_reference.grid.structured_axes[2].point_coordinates
        old_edges = (old_points - old_points[0]) / (old_points[-1] - old_points[0])
        new_edges = (new_points - new_points[0]) / (new_points[-1] - new_points[0])
        old_sigma = 0.5 * (old_edges[:-1] + old_edges[1:])
        new_sigma = 0.5 * (new_edges[:-1] + new_edges[1:])
        velocity = tuple(
            self._interpolate_velocity(
                component,
                old_sigma,
                new_sigma,
                old_edges,
                new_edges,
            )
            for component in old_view.velocity
        )
        new_state = new_hydrodynamics.initial_state(
            old_state.eta,
            velocity=velocity,
        )
        old_column_vertices = self._column_vertical_vertices(old_geometry.mapped_vertices)
        new_column_vertices = self._column_vertical_vertices(new_geometry.mapped_vertices)
        remapped_content = {
            name: self._vertical_overlap_content(
                old_column_vertices,
                new_column_vertices,
                content,
                old_geometry.cell_volumes,
                new_hydrodynamics.surface.horizontal_area,
            )
            for name, content in old_state.scalar_content.items()
        }
        new_state = FreeSurfaceALEState(
            new_state.eta,
            new_state.momentum,
            remapped_content,
        )
        new_continuation = FreeSurfaceALEContinuationState.initialize(new_state)
        new_continuation = FreeSurfaceALEContinuationState(
            new_state,
            continuation.eta_rate,
            new_continuation.pressure_head,
            continuation.ledger,
            continuation.mesh_epoch + 1,
            continuation.wave_controller,
        )
        scalar_defect = {
            name: jnp.sum(remapped_content[name])
            - jnp.sum(old_state.scalar_content[name])
            for name in remapped_content
        }
        old_momentum = sum(jnp.sum(component) for component in old_state.momentum)
        new_momentum = sum(jnp.sum(component) for component in new_state.momentum)
        old_energy = old_view.kinetic_energy
        new_energy = new_hydrodynamics.view(
            new_state, continuation.eta_rate
        ).kinetic_energy
        old_quality = old_surface.geometry_evidence(
            old_state.eta, continuation.eta_rate
        ).minimum_height
        new_quality = new_hydrodynamics.surface.geometry_evidence(
            new_state.eta, continuation.eta_rate
        ).minimum_height
        scalar_norm = max(
            tuple(jnp.abs(value) for value in scalar_defect.values()),
            default=jnp.asarray(0.0),
        )
        finite = (
            jnp.all(jnp.stack(tuple(jnp.isfinite(v) for v in scalar_defect.values())))
            & jnp.isfinite(old_momentum)
            & jnp.isfinite(new_momentum)
            & jnp.isfinite(old_energy)
            & jnp.isfinite(new_energy)
        )
        relative_tolerance = jnp.maximum(
            4096.0 * jnp.finfo(old_state.eta.dtype).eps,
            1.0e-10,
        )
        conservative = scalar_norm <= relative_tolerance * jnp.maximum(
            max(
                tuple(jnp.abs(jnp.sum(v)) for v in old_state.scalar_content.values()),
                default=jnp.asarray(1.0),
            ),
            1.0,
        )
        event_id = canonical_fingerprint(
            {
                "kind": "free-surface-rezone-event",
                "plan": self.plan_id,
                "old": hydrodynamics.prepared_id,
                "new": new_hydrodynamics.prepared_id,
            }
        )
        evidence = RezoneEvidence(
            scalar_content_defect=scalar_defect,
            momentum_defect=new_momentum - old_momentum,
            kinetic_energy_defect=new_energy - old_energy,
            old_quality=old_quality,
            new_quality=new_quality,
            projection_residual=jnp.asarray(0.0, dtype=old_state.eta.dtype),
            finite=finite,
            conservative=conservative,
            successful=finite
            & conservative
            & (new_quality + self.minimum_quality_improvement >= old_quality),
            event_id=event_id,
        )
        return FreeSurfaceRezoneResult(new_hydrodynamics, new_continuation, evidence)


class GraphShorelineEventPlan(StrictModule, NonTrainableState):
    """Detect fixed-graph shoreline limits and issue explicit handoff status."""

    rezone_height: float = eqx.field(static=True)
    dry_height: float = eqx.field(static=True)
    two_phase_slope: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        rezone_height: float = 0.05,
        dry_height: float = 0.005,
        two_phase_slope: float = 0.8,
    ):
        values = tuple(float(v) for v in (rezone_height, dry_height, two_phase_slope))
        if any(not np.isfinite(v) or v <= 0.0 for v in values) or values[1] >= values[0]:
            raise ValueError("Invalid graph shoreline event thresholds.")
        self.rezone_height = values[0]
        self.dry_height = values[1]
        self.two_phase_slope = values[2]
        self.plan_id = canonical_fingerprint(
            {"kind": "graph-shoreline-event-plan", "thresholds": list(values)}
        )

    def evaluate(
        self,
        hydrodynamics: PreparedOnePhaseFreeSurfaceALE,
        continuation: FreeSurfaceALEContinuationState,
        /,
    ) -> ShorelineEvent:
        evidence = hydrodynamics.surface.geometry_evidence(
            continuation.state.eta, continuation.eta_rate
        )
        minimum = evidence.minimum_height
        slope = evidence.maximum_slope
        if not bool(evidence.finite):
            status: ShorelineStatus = "reject"
            target = "none"
            code = 4
        elif float(minimum) <= self.dry_height or float(slope) >= self.two_phase_slope:
            status = "handoff-two-phase"
            target = "incompressible-two-phase-vof"
            code = 3
        elif float(minimum) <= self.rezone_height:
            status = "rezone"
            target = "free-surface-vertical-rezone"
            code = 1
        else:
            status = "continue"
            target = "one-phase-free-surface-ale"
            code = 0
        event_id = canonical_fingerprint(
            {
                "kind": "graph-shoreline-event",
                "plan": self.plan_id,
                "surface": hydrodynamics.surface.surface_id,
                "status": status,
            }
        )
        return ShorelineEvent(
            minimum_height=minimum,
            maximum_slope=slope,
            status_code=jnp.asarray(code, dtype=jnp.int32),
            finite=evidence.finite,
            valid=evidence.valid | (status != "continue"),
            status=status,
            target_product=target,
            event_id=event_id,
        )


__all__ = [
    "FreeSurfaceRezonePlan",
    "FreeSurfaceRezoneResult",
    "GraphShorelineEventPlan",
    "RezoneEvidence",
    "ShorelineEvent",
]
