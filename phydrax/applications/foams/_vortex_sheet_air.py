#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Circulation-preserving vortex-sheet air for explicit foam surfaces."""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum
from typing import final, Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import parameter_field, ParameterOwner
from ..._validation import nonnegative_integer, positive_finite_float, positive_integer
from ...discretization.vortex import VortexSourceState, VortexTargetState
from ...geometry.multiregion_surface import (
    apply_surface_events,
    MultiRegionSurfaceState,
    MultiRegionSurfaceTopology,
    PreparedMultiRegionSurface,
    SurfaceEventPassEvidence,
    SurfaceEventPolicy,
    SurfaceEventProposal,
)
from ...interfacial_transport import SurfaceFilmEvidence
from ...operators.integral.vortex import (
    GaussianErfDirectVortexPlan3D,
    PreparedGaussianErfDirectVortex3D,
    PreparedVortexFMM,
    VortexFMMEvidence,
    VortexFMMPlan,
)
from ...sparse import EdgeRelation, linear_apply, route_reduce
from ...typing import checked
from ._air import FoamAirModel
from ._dynamics import FoamDynamicsState, PreparedFoamDynamics
from ._rupture import apply_foam_rupture, FoamRuptureEvidence, FoamRupturePlan


type VortexSheetCorePolicy = Literal["mean-edge-fraction"]
_CIRCULATION_FIELD = "circulation"


class VortexSheetAirStatus(IntEnum):
    """Outcome of a bounded vortex-sheet advance."""

    COMPLETED = 0
    FMM_FAILED = 1
    FMM_ERROR_EXCEEDED = 2
    CONSTRAINT_FAILED = 3
    CCD_FAILED = 4
    EVENT_ROLLED_BACK = 5
    NONFINITE = 6


class VortexSheetCurvatureRelationStatus(IntEnum):
    """Host preparation outcome of the bounded wedge-to-slot relation."""

    ACCEPTED = 0
    ROUTE_CAPACITY_EXCEEDED = 1


@final
class VortexSheetCurvatureRelationEvidence(StrictModule):
    """Resource evidence for the prepared signed-curvature accumulation."""

    status: VortexSheetCurvatureRelationStatus = eqx.field(static=True)
    accepted: bool = eqx.field(static=True)
    required_routes: int = eqx.field(static=True)
    route_capacity: int = eqx.field(static=True)
    retained_routes: int = eqx.field(static=True)
    target_shape: tuple[int, int, int] = eqx.field(static=True)
    logical_retained_bytes: int = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class VortexSheetCurvatureRelationCapacityError(ValueError):
    """The wedge-to-slot relation exceeded its declared route capacity."""

    def __init__(self, evidence: VortexSheetCurvatureRelationEvidence, /) -> None:
        self.evidence = evidence
        super().__init__(
            "Vortex-sheet curvature relation preparation failed with status "
            f"{evidence.status.name}: required {evidence.required_routes} routes, "
            f"capacity {evidence.route_capacity}."
        )


@final
class VortexSheetAirPlan(StrictModule, ParameterOwner):
    """Vortex-sheet time step, Gaussian core, FMM, and reference budgets.

    The only regularization is the declared Gaussian core
    ``max(minimum_core_radius, core_radius_fraction * mean_face_edge)``.
    Circulation is never smoothed. An explicit ``maximum_curvature_routes``
    bounds the prepared local wedge-to-sheet relation and is refused with
    resource evidence before numerical execution; otherwise the topology
    capacities provide the bound. The FMM reference envelope is reused
    while source-face centroids remain within
    ``maximum_displacement_fraction`` of their prepared positions. A topology
    epoch change requires a new prepared object.
    """

    air_density: Array = parameter_field()
    surface_tension_scale: Array = parameter_field()
    time_step: float = eqx.field(static=True)
    steps: int = eqx.field(static=True)
    core_policy: VortexSheetCorePolicy = eqx.field(static=True)
    core_radius_fraction: float = eqx.field(static=True)
    minimum_core_radius: float = eqx.field(static=True)
    bounding_padding_fraction: float = eqx.field(static=True)
    maximum_displacement_fraction: float = eqx.field(static=True)
    fmm_depth: int = eqx.field(static=True)
    fmm_expansion_order: int = eqx.field(static=True)
    fmm_leaf_capacity: int = eqx.field(static=True)
    maximum_fmm_relative_error: float = eqx.field(static=True)
    direct_reference_maximum_interactions: int = eqx.field(static=True)
    maximum_curvature_routes: int | None = eqx.field(static=True)
    model: FoamAirModel = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        air_density: float,
        surface_tension_scale: float = 1.0,
        time_step: float,
        steps: int = 1,
        core_radius_fraction: float = 0.35,
        minimum_core_radius: float = 1.0e-12,
        bounding_padding_fraction: float = 0.2,
        maximum_displacement_fraction: float = 0.1,
        fmm_depth: int = 3,
        fmm_expansion_order: int = 1,
        fmm_leaf_capacity: int = 64,
        maximum_fmm_relative_error: float = 0.1,
        direct_reference_maximum_interactions: int = 1_000_000,
        maximum_curvature_routes: int | None = None,
    ) -> None:
        density = positive_finite_float(air_density, "air_density")
        tension_scale = float(surface_tension_scale)
        if not np.isfinite(tension_scale) or tension_scale < 0.0:
            raise ValueError("surface_tension_scale must be finite and nonnegative.")
        step = positive_finite_float(time_step, "time_step")
        count = positive_integer(steps, "steps")
        core_fraction = positive_finite_float(
            core_radius_fraction, "core_radius_fraction"
        )
        minimum_core = positive_finite_float(minimum_core_radius, "minimum_core_radius")
        padding = positive_finite_float(
            bounding_padding_fraction, "bounding_padding_fraction"
        )
        displacement = positive_finite_float(
            maximum_displacement_fraction, "maximum_displacement_fraction"
        )
        depth = positive_integer(fmm_depth, "fmm_depth")
        order = nonnegative_integer(fmm_expansion_order, "fmm_expansion_order")
        leaf = positive_integer(fmm_leaf_capacity, "fmm_leaf_capacity")
        error = positive_finite_float(
            maximum_fmm_relative_error, "maximum_fmm_relative_error"
        )
        direct = positive_integer(
            direct_reference_maximum_interactions,
            "direct_reference_maximum_interactions",
        )
        curvature_routes = (
            None
            if maximum_curvature_routes is None
            else positive_integer(
                maximum_curvature_routes, "maximum_curvature_routes"
            )
        )
        if core_fraction > 2.0:
            raise ValueError("core_radius_fraction must not exceed two.")
        if padding > 2.0 or displacement >= padding:
            raise ValueError("FMM fractions require displacement < padding <= 2.")
        if order not in (0, 1):
            raise ValueError("VortexFMMPlan supports expansion order zero or one.")
        if error >= 1.0:
            raise ValueError("maximum_fmm_relative_error must be below one.")
        self.air_density = jnp.asarray(density, dtype=jnp.float64)
        self.surface_tension_scale = jnp.asarray(tension_scale, dtype=jnp.float64)
        self.time_step = step
        self.steps = count
        self.core_policy = "mean-edge-fraction"
        self.core_radius_fraction = core_fraction
        self.minimum_core_radius = minimum_core
        self.bounding_padding_fraction = padding
        self.maximum_displacement_fraction = displacement
        self.fmm_depth = depth
        self.fmm_expansion_order = order
        self.fmm_leaf_capacity = leaf
        self.maximum_fmm_relative_error = error
        self.maximum_curvature_routes = curvature_routes
        self.direct_reference_maximum_interactions = direct
        self.model = FoamAirModel.VORTEX_SHEET
        content = {
            "kind": "vortex-sheet-air-plan",
            "air_density": float(density).hex(),
            "surface_tension_scale": tension_scale.hex(),
            "time_step": float(step).hex(),
            "steps": count,
            "core_policy": self.core_policy,
            "core_radius_fraction": float(core_fraction).hex(),
            "minimum_core_radius": float(minimum_core).hex(),
            "bounding_padding_fraction": float(padding).hex(),
            "maximum_displacement_fraction": float(displacement).hex(),
            "fmm_depth": depth,
            "fmm_expansion_order": order,
            "fmm_leaf_capacity": leaf,
            "maximum_fmm_relative_error": float(error).hex(),
            "direct_reference_maximum_interactions": direct,
        }
        if curvature_routes is not None:
            content["maximum_curvature_routes"] = curvature_routes
        self.plan_id = canonical_fingerprint(content)

    def prepare(
        self,
        dynamics: PreparedFoamDynamics,
        state: FoamDynamicsState,
        /,
    ) -> PreparedVortexSheetAir:
        """Prepare the fixed-topology FMM and E6 constrained integrator."""
        return PreparedVortexSheetAir(self, dynamics, state)


@final
class VortexSheetAirState(StrictModule):
    """E6 mechanics state plus intensive circulation on every sheet slot."""

    dynamics: FoamDynamicsState
    circulation: Array
    removed_circulation: Array

    @checked
    def __init__(
        self,
        dynamics: FoamDynamicsState,
        circulation: ArrayLike,
        /,
        *,
        removed_circulation: ArrayLike = 0.0,
    ) -> None:
        values = jnp.asarray(circulation, dtype=dynamics.surface.positions.dtype)
        expected = (
            dynamics.surface.vertex_capacity,
            dynamics.surface.slot_width,
        )
        if values.shape != expected:
            raise ValueError(f"circulation must have sheet-slot shape {expected}.")
        values = eqx.error_if(
            values,
            jnp.any(
                jnp.where(dynamics.surface.slot_active, ~jnp.isfinite(values), False)
            ),
            "Active circulation values must be finite.",
        )
        removed = jnp.asarray(removed_circulation, dtype=dynamics.surface.positions.dtype)
        if removed.ndim != 0:
            raise ValueError("removed_circulation must be scalar.")
        removed = eqx.error_if(
            removed,
            ~jnp.isfinite(removed),
            "removed_circulation must be one finite scalar.",
        )
        self.dynamics = dynamics
        self.circulation = jnp.where(
            dynamics.surface.slot_active, values, jnp.zeros_like(values)
        )
        self.removed_circulation = removed

    @property
    def surface(self) -> MultiRegionSurfaceState:
        return self.dynamics.surface


@final
class VortexSheetAirEvidence(StrictModule):
    """Gauge, FMM, regularization, energy, constraint, and epoch evidence."""

    status: Array
    accepted: Array
    circulation_pair_means: Array
    maximum_gauge_residual: Array
    circulation_content_before: Array
    circulation_content_after: Array
    circulation_update_residual: Array
    circulation_conservation_residual: Array
    circulation_transfer_residual: Array
    fmm_successful: Array
    fmm_geometric_tail_bound: Array
    fmm_maximum_reference_displacement: Array
    fmm_stale_topology: Array
    fmm_source_overflow: Array
    fmm_interaction_count: Array
    fmm_source_count: Array
    fmm_target_count: Array
    direct_reference_evaluated: Array
    fmm_relative_l2_error: Array
    minimum_core_radius: Array
    maximum_core_radius: Array
    initial_kinetic_energy: Array
    final_kinetic_energy: Array
    initial_surface_energy: Array
    final_surface_energy: Array
    surface_work: Array
    projection_work: Array
    energy_work_residual: Array
    volume_residual: Array
    constraint_rank: Array
    constraint_condition: Array
    ccd_certified: Array
    minimum_ccd_time_of_impact: Array
    finite: Array
    topology_changed: Array
    rebuild_required: Array
    event: SurfaceEventPassEvidence | None
    no_circulation_smoothing: bool = eqx.field(static=True)
    core_policy: VortexSheetCorePolicy = eqx.field(static=True)
    source_epoch: Array
    target_epoch: Array
    derivative_available: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


@final
class VortexSheetAirResult(StrictModule):
    """Committed vortex-sheet state, induced velocity, and complete evidence."""

    topology: MultiRegionSurfaceTopology
    state: VortexSheetAirState
    velocity: Array
    intrinsic_gradient: Array
    sheet_strength: Array
    circulation_source: Array
    evidence: VortexSheetAirEvidence

    @property
    def successful(self) -> bool:
        return int(self.evidence.status) == VortexSheetAirStatus.COMPLETED


def _slot_pair_relation(topology: MultiRegionSurfaceTopology, /) -> EdgeRelation:
    slots = np.asarray(topology.vertex_pair_slots, dtype=np.int64).reshape(-1)
    active = np.asarray(topology.slot_active).reshape(-1)
    sources = np.flatnonzero(active)
    return EdgeRelation(
        sources,
        slots[sources],
        source_size=slots.size,
        target_size=topology.region_pair_capacity,
    )


def _curvature_slot_relation(
    topology: MultiRegionSurfaceTopology,
    maximum_routes: int,
    /,
) -> tuple[EdgeRelation, Array, VortexSheetCurvatureRelationEvidence]:
    """Prepare local wedge candidates for ``(vertex, slot, side)`` targets."""
    edges = np.asarray(topology.edges, dtype=np.int64)
    wedge_valid = (np.asarray(topology.edge_faces) >= 0) & np.asarray(
        topology.edge_active
    )[:, None]
    slot_active = np.asarray(topology.slot_active, dtype=np.bool_)
    slots = np.asarray(topology.vertex_pair_slots, dtype=np.int64)
    pairs = np.asarray(topology.region_pairs)
    region_index_bytes = pairs.dtype.itemsize
    required = sum(
        2
        * int(np.count_nonzero(wedge_valid[edge_index]))
        * sum(int(np.count_nonzero(slot_active[vertex])) for vertex in edges[edge_index])
        for edge_index in range(topology.edge_count)
    )
    accepted = required <= maximum_routes
    retained = required if accepted else 0
    target_shape = (topology.vertex_capacity, topology.slot_width, 2)
    status = (
        VortexSheetCurvatureRelationStatus.ACCEPTED
        if accepted
        else VortexSheetCurvatureRelationStatus.ROUTE_CAPACITY_EXCEEDED
    )
    evidence_id = canonical_fingerprint(
        {
            "kind": "vortex-sheet-curvature-relation-evidence",
            "status": int(status),
            "topology": topology.topology_id,
            "required_routes": required,
            "route_capacity": maximum_routes,
            "target_shape": list(target_shape),
        }
    )
    evidence = VortexSheetCurvatureRelationEvidence(
        status=status,
        accepted=accepted,
        required_routes=required,
        route_capacity=maximum_routes,
        retained_routes=retained,
        target_shape=target_shape,
        logical_retained_bytes=retained
        * (
            2 * np.dtype(np.int32).itemsize
            + np.dtype(np.bool_).itemsize
            + region_index_bytes
        ),
        topology_id=topology.topology_id,
        evidence_id=evidence_id,
    )
    if not accepted:
        raise VortexSheetCurvatureRelationCapacityError(evidence)
    source_routes: list[int] = []
    target_routes: list[int] = []
    route_regions: list[int] = []
    slot_width = topology.slot_width
    for edge_index in range(topology.edge_count):
        for wedge_index in np.flatnonzero(wedge_valid[edge_index]):
            source = edge_index * topology.valence_width + int(wedge_index)
            for vertex in edges[edge_index]:
                for slot in np.flatnonzero(slot_active[vertex]):
                    target = 2 * (int(vertex) * slot_width + int(slot))
                    pair = pairs[slots[vertex, slot]]
                    source_routes.extend((source, source))
                    target_routes.extend((target, target + 1))
                    route_regions.extend((int(pair[0]), int(pair[1])))
    return (
        EdgeRelation(
            np.asarray(source_routes, dtype=np.int64),
            np.asarray(target_routes, dtype=np.int64),
            source_size=topology.edge_capacity * topology.valence_width,
            target_size=topology.vertex_capacity * topology.slot_width * 2,
        ),
        jnp.asarray(np.asarray(route_regions, dtype=pairs.dtype)),
        evidence,
    )


def _strip_last_sheet_field(
    topology: MultiRegionSurfaceTopology,
    state: MultiRegionSurfaceState,
    /,
) -> MultiRegionSurfaceState:
    return MultiRegionSurfaceState(
        topology,
        state.positions,
        velocities=state.velocities,
        sheet_fields=state.sheet_fields[..., :-1],
        region_fields=state.region_fields,
        sheet_field_names=state.sheet_field_names[:-1],
        region_field_names=state.region_field_names,
    )


@final
class PreparedVortexSheetAir(StrictModule):
    """Prepared FMM structure and E6 constrained mechanics for one topology epoch."""

    plan: VortexSheetAirPlan
    dynamics: PreparedFoamDynamics
    fmm: PreparedVortexFMM
    direct: PreparedGaussianErfDirectVortex3D | None
    slot_pairs: EdgeRelation
    slot_pair_counts: Array
    pair_tension: Array
    curvature_slots: EdgeRelation
    curvature_route_regions: Array
    curvature_relation_evidence: VortexSheetCurvatureRelationEvidence
    source_count: int = eqx.field(static=True)
    target_count: int = eqx.field(static=True)
    topology_epoch: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        plan: VortexSheetAirPlan,
        dynamics: PreparedFoamDynamics,
        state: FoamDynamicsState,
        /,
    ) -> None:
        topology = dynamics.surface.topology
        state.surface.require_topology(topology)
        if dynamics.air.route != "incompressible" or state.gas is not None:
            raise ValueError(
                "Vortex-sheet air currently requires E6 incompressible target volumes."
            )
        points = np.asarray(
            state.surface.positions[: topology.vertex_count], dtype=np.float64
        )
        faces = topology.host_faces()
        centroids = np.mean(points[faces], axis=1)
        combined = np.concatenate((points, centroids), axis=0)
        lower_data = np.min(combined, axis=0)
        upper_data = np.max(combined, axis=0)
        span = float(np.max(upper_data - lower_data))
        if not np.isfinite(span) or span <= 0.0:
            raise ValueError("Vortex-sheet preparation needs nondegenerate geometry.")
        padding = plan.bounding_padding_fraction * span
        lower = lower_data - padding
        upper = upper_data + padding
        displacement = plan.maximum_displacement_fraction * span
        curvature_capacity = (
            4
            * topology.edge_capacity
            * topology.valence_width
            * topology.slot_width
            if plan.maximum_curvature_routes is None
            else plan.maximum_curvature_routes
        )
        (
            curvature_slots,
            curvature_route_regions,
            curvature_evidence,
        ) = _curvature_slot_relation(topology, curvature_capacity)
        fmm_plan = VortexFMMPlan(
            centroids,
            lower,
            upper,
            reference_targets=points,
            depth=plan.fmm_depth,
            expansion_order=plan.fmm_expansion_order,
            leaf_capacity=min(plan.fmm_leaf_capacity, topology.face_count),
            target_leaf_capacity=min(plan.fmm_leaf_capacity, topology.vertex_count),
            maximum_reference_displacement=displacement,
        )
        fmm = fmm_plan.prepare(
            source_capacity=topology.face_count,
            target_capacity=topology.vertex_count,
            target_topology="arbitrary-targets",
        )
        interactions = topology.face_count * topology.vertex_count
        direct: PreparedGaussianErfDirectVortex3D | None = None
        if interactions <= plan.direct_reference_maximum_interactions:
            direct = GaussianErfDirectVortexPlan3D(
                maximum_sources=topology.face_count,
                maximum_targets=topology.vertex_count,
                maximum_interactions=plan.direct_reference_maximum_interactions,
            ).prepare(
                source_capacity=topology.face_count,
                target_capacity=topology.vertex_count,
                target_topology="arbitrary-targets",
            )
        relation = _slot_pair_relation(topology)
        counts = linear_apply(
            relation,
            jnp.ones(relation.route_shape, dtype=jnp.float64),
            jnp.ones(
                (topology.vertex_capacity * topology.slot_width,), dtype=jnp.float64
            ),
        )
        pair_tension_sum = linear_apply(
            dynamics.surface.face_pair_relation,
            jnp.ones(dynamics.surface.face_pair_relation.route_shape, dtype=jnp.float64),
            dynamics.face_tension,
        )
        pair_face_count = linear_apply(
            dynamics.surface.face_pair_relation,
            jnp.ones(dynamics.surface.face_pair_relation.route_shape, dtype=jnp.float64),
            jnp.ones((topology.face_capacity,), dtype=jnp.float64),
        )
        self.plan = plan
        self.dynamics = dynamics
        self.fmm = fmm
        self.direct = direct
        self.curvature_slots = curvature_slots
        self.curvature_route_regions = curvature_route_regions
        self.curvature_relation_evidence = curvature_evidence
        self.slot_pairs = relation
        self.slot_pair_counts = counts
        self.pair_tension = pair_tension_sum / jnp.maximum(pair_face_count, 1.0)
        self.source_count = topology.face_count
        self.target_count = topology.vertex_count
        self.topology_epoch = topology.epoch
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-vortex-sheet-air",
                "plan": plan.plan_id,
                "dynamics": dynamics.prepared_id,
                "topology": topology.topology_id,
                "epoch": topology.epoch,
                "fmm": fmm.prepared_id,
                "direct_reference": None if direct is None else direct.prepared_id,
            }
        )

    @property
    def topology(self) -> MultiRegionSurfaceTopology:
        return self.dynamics.surface.topology

    def gauge(self, circulation: ArrayLike, /) -> tuple[Array, Array]:
        """Subtract the arithmetic mean independently on every region-pair sheet."""
        values = jnp.asarray(circulation)
        shape = (self.topology.vertex_capacity, self.topology.slot_width)
        if values.shape != shape:
            raise ValueError(f"circulation must have sheet-slot shape {shape}.")
        flat = jnp.where(self.topology.slot_active, values, 0.0).reshape(-1)
        sums = linear_apply(
            self.slot_pairs,
            jnp.ones(self.slot_pairs.route_shape, dtype=flat.dtype),
            flat,
        )
        means = sums / jnp.maximum(self.slot_pair_counts.astype(flat.dtype), 1.0)
        pairs = jnp.maximum(self.topology.vertex_pair_slots, 0)
        gauged = jnp.where(
            self.topology.slot_active,
            values - means[pairs],
            0.0,
        )
        check = linear_apply(
            self.slot_pairs,
            jnp.ones(self.slot_pairs.route_shape, dtype=flat.dtype),
            gauged.reshape(-1),
        ) / jnp.maximum(self.slot_pair_counts.astype(flat.dtype), 1.0)
        return gauged, check

    def initialize(
        self,
        dynamics: FoamDynamicsState,
        circulation: ArrayLike | None = None,
        /,
    ) -> VortexSheetAirState:
        """Create a gauge-fixed sheet state on this prepared epoch."""
        dynamics.surface.require_topology(self.topology)
        values = (
            jnp.zeros(
                (self.topology.vertex_capacity, self.topology.slot_width),
                dtype=dynamics.surface.positions.dtype,
            )
            if circulation is None
            else jnp.asarray(circulation, dtype=dynamics.surface.positions.dtype)
        )
        gauged, _ = self.gauge(values)
        return VortexSheetAirState(dynamics, gauged)

    def _face_geometry(
        self, positions: Array, /
    ) -> tuple[Array, Array, Array, Array, Array]:
        topology = self.topology
        faces = jnp.maximum(topology.faces, 0)
        corners = positions[faces]
        first = corners[:, 1] - corners[:, 0]
        second = corners[:, 2] - corners[:, 0]
        twice_area_vector = jnp.cross(first, second)
        twice_area = jnp.linalg.norm(twice_area_vector, axis=1)
        safe = jnp.where(topology.face_active, twice_area, 1.0)
        normal = jnp.where(
            topology.face_active[:, None],
            twice_area_vector / safe[:, None],
            0.0,
        )
        canonical_normal = normal * topology.face_pair_signs[:, None]
        edge_lengths = jnp.stack(
            (
                jnp.linalg.norm(corners[:, 1] - corners[:, 0], axis=1),
                jnp.linalg.norm(corners[:, 2] - corners[:, 1], axis=1),
                jnp.linalg.norm(corners[:, 0] - corners[:, 2], axis=1),
            ),
            axis=1,
        )
        core = jnp.maximum(
            self.plan.minimum_core_radius,
            self.plan.core_radius_fraction * jnp.mean(edge_lengths, axis=1),
        )
        core = jnp.where(topology.face_active, core, self.plan.minimum_core_radius)
        return corners, 0.5 * twice_area, normal, canonical_normal, core

    def _wedge_half_curvature(
        self, positions: ArrayLike, /
    ) -> tuple[Array, Array, Array, Array]:
        points = jnp.asarray(positions)
        expected = (self.topology.vertex_capacity, 3)
        if points.shape != expected:
            raise ValueError(f"positions must have shape {expected}.")
        wedges = self.dynamics.surface.junction_wedges(points)
        edges = jnp.maximum(self.topology.edges, 0)
        edge_length = jnp.linalg.norm(
            points[edges[:, 1]] - points[edges[:, 0]], axis=1
        )
        half_curvature = 0.5 * edge_length[:, None] * (jnp.pi - wedges.angles)
        return points, half_curvature, wedges.regions, wedges.valid

    def _slot_integrated_signed_curvature(self, positions: ArrayLike, /) -> Array:
        points, half_curvature, wedge_regions, _ = self._wedge_half_curvature(positions)
        relation = self.curvature_slots
        safe_source = jnp.where(relation.valid, relation.source_indices, 0)
        coefficients = (
            relation.valid
            & (
                jnp.maximum(wedge_regions, 0).reshape(-1)[safe_source]
                == self.curvature_route_regions
            )
        ).astype(points.dtype)
        accumulated = linear_apply(
            relation,
            coefficients,
            half_curvature.reshape(-1),
        )
        return accumulated.reshape(
            (self.topology.vertex_capacity, self.topology.slot_width, 2)
        )

    def integrated_signed_curvature(self, positions: ArrayLike, /) -> Array:
        """Signed edge curvature assigned to each ``(vertex, region)``.

        Da et al. (2015), equations (10)--(11), assign half of every
        ``|e| theta_i`` edge measure to each endpoint.  A region's signed
        turning angle is ``pi - wedge_angle``: manifold wedges are
        ``pi +/- bend``, and all three values agree at a 120-degree Plateau
        border.

        This explicit dense diagnostic preserves the public ``(vertex, region)``
        result. Fixed-topology circulation steps use the prepared local
        ``(vertex, slot, side)`` relation and do not materialize this array.
        """
        points, half_curvature, wedge_regions, wedge_valid = (
            self._wedge_half_curvature(positions)
        )
        edges = jnp.maximum(self.topology.edges, 0)
        shape = (
            self.topology.edge_capacity,
            2,
            self.topology.valence_width,
        )
        vertices = jnp.broadcast_to(edges[:, :, None], shape)
        regions = jnp.broadcast_to(jnp.maximum(wedge_regions, 0)[:, None, :], shape)
        values = jnp.broadcast_to(half_curvature[:, None, :], shape)
        valid = jnp.broadcast_to(wedge_valid[:, None, :], shape)
        return (
            jnp.zeros(
                (self.topology.vertex_capacity, self.topology.region_capacity),
                dtype=points.dtype,
            )
            .at[vertices, regions]
            .add(jnp.where(valid, values, 0.0))
        )

    def circulation_source(self, state: VortexSheetAirState, /) -> Array:
        """Source-faithful signed-curvature circulation rate.

        ``Gamma`` uses the normal out of the canonical pair's first region,
        opposite to Da et al.'s higher-to-lower normal.  Their
        ``-sigma (H_i-H_j)/(rho A_v)`` therefore has the positive sign below.
        ``pair_tension = 2 sigma`` is the effective two-interface film tension.
        """
        state.surface.require_topology(self.topology)
        positions = state.surface.positions
        curvature = self._slot_integrated_signed_curvature(positions)
        pair_slots = jnp.maximum(self.topology.vertex_pair_slots, 0)
        first = curvature[..., 0]
        second = curvature[..., 1]
        slot_areas = self.dynamics.surface.slot_areas(positions)
        safe_area = jnp.where(self.topology.slot_active, slot_areas, 1.0)
        raw = (
            self.plan.surface_tension_scale
            * (0.5 * self.pair_tension[pair_slots])
            * (first - second)
            / (self.plan.air_density * safe_area)
        )
        source, _ = self.gauge(raw)
        return source

    def intrinsic_sheet_gradient(
        self, circulation: ArrayLike, positions: ArrayLike, /
    ) -> tuple[Array, Array]:
        """Piecewise-linear intrinsic gradient and ``gamma = n x grad Gamma``."""
        values = jnp.asarray(circulation)
        points = jnp.asarray(positions)
        corners, _, normal, canonical_normal, _ = self._face_geometry(points)
        faces = jnp.maximum(self.topology.faces, 0)
        slots = jnp.maximum(self.topology.face_corner_slots, 0)
        gamma_corner = values[faces, slots]
        opposite = jnp.stack(
            (
                corners[:, 2] - corners[:, 1],
                corners[:, 0] - corners[:, 2],
                corners[:, 1] - corners[:, 0],
            ),
            axis=1,
        )
        twice_area = jnp.linalg.norm(
            jnp.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
            axis=1,
        )
        basis_gradient = jnp.cross(normal[:, None, :], opposite) / jnp.maximum(
            twice_area[:, None, None], jnp.finfo(points.dtype).tiny
        )
        gradient = jnp.sum(gamma_corner[..., None] * basis_gradient, axis=1)
        gradient = jnp.where(self.topology.face_active[:, None], gradient, 0.0)
        strength = jnp.cross(canonical_normal, gradient)
        return gradient, strength

    def _source_states(
        self, circulation: Array, positions: Array, /
    ) -> tuple[VortexSourceState, VortexTargetState, Array, Array, Array]:
        gradient, sheet_strength = self.intrinsic_sheet_gradient(circulation, positions)
        _, areas, _, _, core = self._face_geometry(positions)
        faces = jnp.maximum(self.topology.faces[: self.source_count], 0)
        source_positions = jnp.mean(positions[faces], axis=1)
        source = VortexSourceState(
            source_positions,
            sheet_strength[: self.source_count] * areas[: self.source_count, None],
            core_radius=core[: self.source_count],
        )
        target = VortexTargetState(positions[: self.target_count])
        return source, target, gradient, sheet_strength, core

    def _evaluate_velocity(
        self,
        circulation: Array,
        positions: Array,
        /,
        *,
        use_direct: bool = False,
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array, VortexFMMEvidence]:
        if use_direct and self.direct is None:
            raise ValueError(
                "Direct vortex evaluation exceeds the prepared interaction budget."
            )
        source, target, gradient, sheet_strength, core = self._source_states(
            circulation, positions
        )
        evaluated = self.fmm.evaluate(source, target)
        if evaluated.velocity is None:
            raise RuntimeError("Vortex FMM did not return the requested velocity.")
        selected_velocity = evaluated.velocity
        selected_successful = evaluated.successful
        direct_evaluated = jnp.asarray(self.direct is not None)
        relative_error = jnp.asarray(jnp.nan, dtype=positions.dtype)
        if self.direct is not None:
            reference = self.direct.evaluate(source, target)
            if reference.velocity is None:
                raise RuntimeError("Direct vortex reference did not return velocity.")
            difference = jnp.linalg.norm(evaluated.velocity - reference.velocity)
            scale = jnp.maximum(
                jnp.linalg.norm(reference.velocity),
                jnp.finfo(positions.dtype).tiny,
            )
            relative_error = difference / scale
            if use_direct:
                selected_velocity = reference.velocity
                selected_successful = reference.successful
        velocity = (
            jnp.zeros_like(positions).at[: self.target_count].set(selected_velocity)
        )
        backend = evaluated.diagnostics.backend_diagnostics
        interactions = evaluated.diagnostics.active_interaction_count
        return (
            velocity,
            gradient,
            sheet_strength,
            core,
            selected_successful,
            direct_evaluated,
            relative_error,
            interactions,
            backend,
        )

    def bounded_direct_velocity(
        self, circulation: ArrayLike, positions: ArrayLike, /
    ) -> tuple[Array, Array]:
        """Evaluate the explicit bounded direct route without FMM fallback."""
        values = jnp.asarray(circulation)
        points = jnp.asarray(positions)
        circulation_shape = (
            self.topology.vertex_capacity,
            self.topology.slot_width,
        )
        position_shape = (self.topology.vertex_capacity, 3)
        if values.shape != circulation_shape:
            raise ValueError(f"circulation must have shape {circulation_shape}.")
        if points.shape != position_shape:
            raise ValueError(f"positions must have shape {position_shape}.")
        if self.direct is None:
            raise ValueError(
                "Direct vortex evaluation exceeds the prepared interaction budget."
            )
        source, target, _, _, _ = self._source_states(values, points)
        evaluated = self.direct.evaluate(source, target)
        if evaluated.velocity is None:
            raise RuntimeError("Direct vortex evaluator did not return velocity.")
        velocity = jnp.zeros_like(points).at[: self.target_count].set(evaluated.velocity)
        return velocity, evaluated.successful

    def _kinetic_energy(self, positions: Array, velocity: Array, core: Array, /) -> Array:
        areas = self.dynamics.surface.face_areas(positions)
        route_values = jnp.repeat((areas * core / 3.0)[:, None], 3, axis=0)
        slot_volume = route_reduce(
            self.dynamics.surface.corner_slots,
            route_values,
        ).reshape((self.topology.vertex_capacity, self.topology.slot_width))
        vertex_volume = jnp.sum(slot_volume, axis=1)
        return (
            0.5
            * self.plan.air_density
            * jnp.sum(
                jnp.where(
                    self.topology.vertex_active,
                    vertex_volume * jnp.sum(velocity * velocity, axis=1),
                    0.0,
                )
            )
        )

    @checked
    def fixed_topology_step(self, state: VortexSheetAirState, /) -> VortexSheetAirResult:
        """One symplectic-Euler circulation/FMM/E6 constrained step."""
        state.surface.require_topology(self.topology)
        if state.surface.epoch != self.topology_epoch:
            raise ValueError(
                "Vortex-sheet topology epoch changed; rebuild the prepared plan."
            )
        positions = state.surface.positions
        slot_areas_before = self.dynamics.surface.slot_areas(positions)
        content_before = jnp.sum(
            jnp.where(
                self.topology.slot_active,
                slot_areas_before * state.circulation,
                0.0,
            )
        )
        source = self.circulation_source(state)
        updated, pair_means = self.gauge(state.circulation + self.plan.time_step * source)
        proposed_circulation = jnp.where(
            self.plan.surface_tension_scale == 0.0,
            state.circulation,
            updated,
        )
        (
            velocity,
            gradient,
            sheet_strength,
            core,
            fmm_successful,
            direct_evaluated,
            relative_error,
            interactions,
            backend,
        ) = self._evaluate_velocity(proposed_circulation, positions)
        error_ok = (~direct_evaluated) | (
            relative_error <= self.plan.maximum_fmm_relative_error
        )
        kinematic = self.dynamics.constrained_kinematic_step(
            state.dynamics,
            velocity,
            jnp.asarray(self.plan.time_step, dtype=positions.dtype),
        )
        ccd_certified = False
        impact = 0.0
        if bool(kinematic.accepted) and bool(fmm_successful & error_ok):
            ccd_certified, impact = self.dynamics.certify_motion(
                positions, kinematic.state.surface.positions
            )
        ccd = jnp.asarray(ccd_certified)
        accepted = fmm_successful & error_ok & kinematic.accepted & ccd
        dynamics = FoamDynamicsState(
            MultiRegionSurfaceState(
                self.topology,
                jnp.where(
                    accepted,
                    kinematic.state.surface.positions,
                    state.surface.positions,
                ),
                velocities=jnp.where(
                    accepted,
                    kinematic.state.surface.velocities,
                    state.surface.velocities,
                ),
                sheet_fields=state.surface.sheet_fields,
                region_fields=state.surface.region_fields,
                sheet_field_names=state.surface.sheet_field_names,
                region_field_names=state.surface.region_field_names,
            ),
            unresolved_rim_content=state.dynamics.unresolved_rim_content,
            time=jnp.where(accepted, kinematic.state.time, state.dynamics.time),
        )
        circulation = jnp.where(accepted, proposed_circulation, state.circulation)
        candidate = VortexSheetAirState(
            dynamics,
            circulation,
            removed_circulation=state.removed_circulation,
        )
        final_positions = candidate.surface.positions
        slot_areas_after = self.dynamics.surface.slot_areas(final_positions)
        content_after = jnp.sum(
            jnp.where(
                self.topology.slot_active,
                slot_areas_after * candidate.circulation,
                0.0,
            )
        )
        initial_surface = self.dynamics.surface.surface_energy(
            positions, self.dynamics.face_tension
        )
        final_surface = self.dynamics.surface.surface_energy(
            final_positions, self.dynamics.face_tension
        )
        initial_kinetic = self._kinetic_energy(positions, state.surface.velocities, core)
        final_kinetic = self._kinetic_energy(
            final_positions, candidate.surface.velocities, core
        )
        surface_work = initial_surface - final_surface
        projection_work = kinematic.mechanics.projection_work
        energy_residual = final_kinetic - initial_kinetic - surface_work - projection_work
        expected_circulation = jnp.where(
            self.plan.surface_tension_scale == 0.0,
            state.circulation,
            updated,
        )
        targets = self.dynamics.air.target_volumes
        if targets is None:
            raise RuntimeError("Vortex-sheet air lost its incompressible targets.")
        volumes = self.dynamics.surface.region_volumes(final_positions)[
            self.dynamics.finite_slots
        ]
        volume_residual = jnp.max(
            jnp.abs(volumes - targets)
            / jnp.maximum(jnp.abs(targets), jnp.finfo(positions.dtype).tiny)
        )
        update_residual = jnp.max(
            jnp.abs(
                jnp.where(
                    self.topology.slot_active,
                    proposed_circulation - expected_circulation,
                    0.0,
                )
            )
        )
        conservation_residual = jnp.where(
            self.plan.surface_tension_scale == 0.0,
            jnp.max(jnp.abs(candidate.circulation - state.circulation)),
            update_residual,
        )
        finite = (
            jnp.all(jnp.isfinite(candidate.circulation))
            & jnp.all(jnp.isfinite(velocity))
            & jnp.isfinite(final_surface)
            & jnp.isfinite(final_kinetic)
        )
        status = jnp.where(
            ~finite,
            int(VortexSheetAirStatus.NONFINITE),
            jnp.where(
                ~fmm_successful,
                int(VortexSheetAirStatus.FMM_FAILED),
                jnp.where(
                    ~error_ok,
                    int(VortexSheetAirStatus.FMM_ERROR_EXCEEDED),
                    jnp.where(
                        ~kinematic.accepted,
                        int(VortexSheetAirStatus.CONSTRAINT_FAILED),
                        jnp.where(
                            ~ccd,
                            int(VortexSheetAirStatus.CCD_FAILED),
                            int(VortexSheetAirStatus.COMPLETED),
                        ),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        evidence = VortexSheetAirEvidence(
            status=status,
            accepted=accepted & finite,
            circulation_conservation_residual=conservation_residual,
            circulation_pair_means=pair_means,
            maximum_gauge_residual=jnp.max(jnp.abs(pair_means)),
            circulation_content_before=content_before,
            circulation_content_after=content_after,
            circulation_update_residual=update_residual,
            circulation_transfer_residual=jnp.asarray(0.0, dtype=positions.dtype),
            fmm_successful=fmm_successful,
            fmm_geometric_tail_bound=backend.geometric_tail_bound,
            fmm_maximum_reference_displacement=backend.maximum_reference_displacement,
            fmm_stale_topology=backend.stale_topology,
            fmm_source_overflow=backend.source_overflow,
            fmm_interaction_count=interactions,
            fmm_source_count=jnp.asarray(self.source_count, dtype=jnp.int32),
            fmm_target_count=jnp.asarray(self.target_count, dtype=jnp.int32),
            direct_reference_evaluated=direct_evaluated,
            fmm_relative_l2_error=relative_error,
            minimum_core_radius=jnp.min(core[: self.source_count]),
            maximum_core_radius=jnp.max(core[: self.source_count]),
            initial_kinetic_energy=initial_kinetic,
            final_kinetic_energy=final_kinetic,
            initial_surface_energy=initial_surface,
            final_surface_energy=final_surface,
            surface_work=surface_work,
            projection_work=projection_work,
            energy_work_residual=energy_residual,
            volume_residual=volume_residual,
            constraint_rank=kinematic.mechanics.constraint_rank,
            constraint_condition=kinematic.mechanics.constraint_condition,
            ccd_certified=ccd,
            minimum_ccd_time_of_impact=jnp.asarray(impact, dtype=positions.dtype),
            finite=finite,
            topology_changed=jnp.asarray(False),
            rebuild_required=jnp.asarray(False),
            event=None,
            no_circulation_smoothing=True,
            core_policy=self.plan.core_policy,
            source_epoch=jnp.asarray(self.topology_epoch, dtype=jnp.int32),
            target_epoch=jnp.asarray(self.topology_epoch, dtype=jnp.int32),
            derivative_available=False,
            plan_id=self.plan.plan_id,
            prepared_id=self.prepared_id,
        )
        return VortexSheetAirResult(
            self.topology,
            candidate,
            velocity,
            gradient,
            sheet_strength,
            source,
            evidence,
        )

    def _inject_circulation_content(
        self, state: VortexSheetAirState, /
    ) -> MultiRegionSurfaceState:
        if _CIRCULATION_FIELD in state.surface.sheet_field_names:
            raise ValueError(
                "The surface already carries a circulation field; vortex air owns it."
            )
        content = state.circulation * self.dynamics.surface.slot_areas(
            state.surface.positions
        )
        return MultiRegionSurfaceState(
            self.topology,
            state.surface.positions,
            velocities=state.surface.velocities,
            sheet_fields=jnp.concatenate(
                (state.surface.sheet_fields, content[..., None]), axis=-1
            ),
            region_fields=state.surface.region_fields,
            sheet_field_names=(*state.surface.sheet_field_names, _CIRCULATION_FIELD),
            region_field_names=state.surface.region_field_names,
        )

    def _apply_events(
        self,
        result: VortexSheetAirResult,
        events: Sequence[SurfaceEventProposal],
        event_policy: SurfaceEventPolicy | None,
        /,
    ) -> VortexSheetAirResult:
        injected = self._inject_circulation_content(result.state)
        event_result = apply_surface_events(
            self.topology,
            injected,
            events,
            policy=event_policy,
        )
        if not event_result.committed:
            evidence = eqx.tree_at(
                lambda value: (
                    value.status,
                    value.accepted,
                    value.event,
                ),
                result.evidence,
                (
                    jnp.asarray(
                        int(VortexSheetAirStatus.EVENT_ROLLED_BACK), dtype=jnp.int32
                    ),
                    jnp.asarray(False),
                    event_result.evidence,
                ),
            )
            return eqx.tree_at(lambda value: value.evidence, result, evidence)
        target_surface = PreparedMultiRegionSurface(
            event_result.topology, event_result.state
        )
        content = event_result.state.sheet_fields[..., -1]
        areas = target_surface.slot_areas(event_result.state.positions)
        raw = jnp.where(
            event_result.topology.slot_active,
            content / jnp.maximum(areas, jnp.finfo(areas.dtype).tiny),
            0.0,
        )
        relation = _slot_pair_relation(event_result.topology)
        counts = linear_apply(
            relation,
            jnp.ones(relation.route_shape, dtype=raw.dtype),
            jnp.ones((raw.size,), dtype=raw.dtype),
        )
        sums = linear_apply(
            relation,
            jnp.ones(relation.route_shape, dtype=raw.dtype),
            raw.reshape(-1),
        )
        means = sums / jnp.maximum(counts, 1.0)
        pairs = jnp.maximum(event_result.topology.vertex_pair_slots, 0)
        circulation = jnp.where(
            event_result.topology.slot_active,
            raw - means[pairs],
            0.0,
        )
        stripped = _strip_last_sheet_field(event_result.topology, event_result.state)
        dynamics = FoamDynamicsState(
            stripped,
            unresolved_rim_content=result.state.dynamics.unresolved_rim_content,
            time=result.state.dynamics.time,
        )
        state = VortexSheetAirState(
            dynamics,
            circulation,
            removed_circulation=result.state.removed_circulation,
        )
        transfer = event_result.evidence.sheet_transfer
        residual = (
            jnp.asarray(jnp.nan, dtype=areas.dtype)
            if transfer is None
            else transfer.absolute_defect[-1]
        )
        evidence = eqx.tree_at(
            lambda value: (
                value.circulation_transfer_residual,
                value.topology_changed,
                value.rebuild_required,
                value.event,
            ),
            result.evidence,
            (
                residual,
                jnp.asarray(True),
                jnp.asarray(True),
                event_result.evidence,
            ),
        )
        evidence = eqx.tree_at(
            lambda value: value.target_epoch,
            evidence,
            jnp.asarray(event_result.topology.epoch, dtype=jnp.int32),
        )
        return VortexSheetAirResult(
            event_result.topology,
            state,
            state.surface.velocities,
            jnp.zeros((event_result.topology.face_capacity, 3), dtype=areas.dtype),
            jnp.zeros((event_result.topology.face_capacity, 3), dtype=areas.dtype),
            jnp.zeros(
                (
                    event_result.topology.vertex_capacity,
                    event_result.topology.slot_width,
                ),
                dtype=areas.dtype,
            ),
            evidence,
        )

    @checked
    def apply_rupture(
        self,
        state: VortexSheetAirState,
        rupture: FoamRupturePlan,
        thickness_m: ArrayLike,
        film_status: ArrayLike,
        film_evidence: SurfaceFilmEvidence,
        geometry_revision: ArrayLike,
        /,
        *,
        event_policy: SurfaceEventPolicy | None = None,
    ) -> tuple[VortexSheetAirState, MultiRegionSurfaceTopology, FoamRuptureEvidence]:
        """Apply E5 BURST with conservative circulation content and a removed ledger."""
        if rupture.circulation_field_name != _CIRCULATION_FIELD:
            raise ValueError(
                "rupture must use the canonical circulation field name 'circulation'."
            )
        injected = self._inject_circulation_content(state)
        outcome = apply_foam_rupture(
            rupture,
            self.topology,
            injected,
            thickness_m,
            film_status,
            film_evidence,
            geometry_revision,
            state.dynamics.unresolved_rim_content,
            event_policy=event_policy,
        )
        if not outcome.successful:
            return state, self.topology, outcome.evidence
        target_surface = PreparedMultiRegionSurface(outcome.topology, outcome.state)
        index = outcome.state.sheet_field_names.index(_CIRCULATION_FIELD)
        content = outcome.state.sheet_fields[..., index]
        areas = target_surface.slot_areas(outcome.state.positions)
        raw = jnp.where(
            outcome.topology.slot_active,
            content / jnp.maximum(areas, jnp.finfo(areas.dtype).tiny),
            0.0,
        )
        relation = _slot_pair_relation(outcome.topology)
        counts = linear_apply(
            relation,
            jnp.ones(relation.route_shape, dtype=raw.dtype),
            jnp.ones((raw.size,), dtype=raw.dtype),
        )
        sums = linear_apply(
            relation,
            jnp.ones(relation.route_shape, dtype=raw.dtype),
            raw.reshape(-1),
        )
        means = sums / jnp.maximum(counts, 1.0)
        circulation = jnp.where(
            outcome.topology.slot_active,
            raw - means[jnp.maximum(outcome.topology.vertex_pair_slots, 0)],
            0.0,
        )
        stripped = _strip_last_sheet_field(outcome.topology, outcome.state)
        dynamics = FoamDynamicsState(
            stripped,
            unresolved_rim_content=outcome.unresolved_rim_content,
            time=state.dynamics.time,
        )
        target = VortexSheetAirState(
            dynamics,
            circulation,
            removed_circulation=(
                state.removed_circulation + outcome.evidence.removed_circulation
            ),
        )
        return target, outcome.topology, outcome.evidence

    def advance(
        self,
        state: VortexSheetAirState,
        /,
        *,
        events: Sequence[SurfaceEventProposal] = (),
        event_policy: SurfaceEventPolicy | None = None,
    ) -> VortexSheetAirResult:
        """Run bounded fixed-topology steps, then one transactional event pass."""
        current = state
        result: VortexSheetAirResult | None = None
        for _ in range(self.plan.steps):
            result = self.fixed_topology_step(current)
            current = result.state
            if not result.successful:
                break
        if result is None:
            raise RuntimeError("A positive step count produced no vortex-sheet step.")
        if events and result.successful:
            return self._apply_events(result, events, event_policy)
        return result


__all__ = [
    "PreparedVortexSheetAir",
    "VortexSheetAirEvidence",
    "VortexSheetAirPlan",
    "VortexSheetAirResult",
    "VortexSheetAirState",
    "VortexSheetAirStatus",
    "VortexSheetCorePolicy",
    "VortexSheetCurvatureRelationCapacityError",
    "VortexSheetCurvatureRelationEvidence",
    "VortexSheetCurvatureRelationStatus",
]
