#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Conservative Plateau-border drainage coupled to manifold film sheets.

Physical Plateau borders are the valence-three edges of one fixed
multiregion topology epoch.  Each edge is a finite-volume cell storing liquid
volume and surfactant amount.  Its cross-section is volume divided by current
edge length.  Viscous flux follows a one-dimensional hydraulic law

``q = -A^2/(C mu) (dp/ds - rho g.t)``

with an explicit triangular-channel shape factor ``C``.  At every network
vertex a single hydraulic potential is the conductance-weighted incident
value.  This local Schur elimination makes the signed incident flux sum zero;
at a four-border dry-foam vertex it is the quad-point mass and pressure
balance.  No dense pair graph or topology-sized runtime loop is formed.

The B-on-E adapter supplies explicit sheet-boundary half-edge routes.  Positive
route flux moves extensive content from a film sheet slot into a border cell.
Liquid, surfactant, and a separately declared evaporation sink have distinct
ledgers.  Topology events remain nondifferentiable; ``rates`` and ``step`` are
differentiable only on one prepared topology and geometry revision.  Across a
committed event pass, ``reprepare_after_events`` rebuilds the network on the
target epoch and transports border content only by declared rules: retained
borders keep it and split borders share it by child length.  Every other
event is refused rather than redistributed from lineage.
"""

from __future__ import annotations

from enum import IntEnum
from typing import assert_never, final, NoReturn

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import parameter_field, ParameterOwner
from ..._validation import canonical_identifier, nonnegative_integer, positive_integer
from ...discretization import TopologyEpoch, TopologyEpochTransition
from ...geometry.multiregion_surface import (
    ConservativeFieldTransfer,
    multiregion_topology_epoch,
    MultiRegionSurfaceLineage,
    MultiRegionSurfaceState,
    MultiRegionSurfaceTopology,
    PreparedMultiRegionSurface,
    SurfaceEventKind,
    SurfaceEventPassEvidence,
    SurfaceEventPolicy,
)
from ...interfacial_transport import PreparedFilmSheetSlots, SurfaceFilmEvidence
from ...sparse import EdgeRelation, gather_routes, route_reduce
from ...typing import ConvertibleToArray
from ._rupture import apply_foam_rupture, FoamRupturePlan, FoamRuptureResult


class PlateauBorderPreparationStatus(IntEnum):
    """Host preparation outcome of one fixed Plateau-border network."""

    ACCEPTED = 0
    TOPOLOGY_MISMATCH = 1
    NO_PHYSICAL_BORDERS = 2
    BORDER_CAPACITY_EXCEEDED = 3
    QUAD_POINT_CAPACITY_EXCEEDED = 4
    INCOMPLETE_SHEET_SUPPORT = 5


class PlateauBorderStatus(IntEnum):
    """Outcome of one fixed-topology border step."""

    ACCEPTED = 0
    INADMISSIBLE_INPUT = 1
    GEOMETRY_REVISION_MISMATCH = 2
    UNSUPPORTED_BOUNDARY_FLUX = 3
    EVAPORATION_UNDECLARED = 4
    COURANT_LIMIT = 5
    NEGATIVE_CONTENT = 6
    NONFINITE = 7


class PlateauBorderRimStatus(IntEnum):
    """Resolution of E5's burst ledger onto physical rim/border support."""

    RESOLVED = 0
    RUPTURE_NOT_COMMITTED = 1
    BORDER_SUPPORT_UNAVAILABLE = 2


@final
class PlateauBorderPreparationEvidence(StrictModule):
    """Capacity and B-on-E support evidence of a prepared border network."""

    status: PlateauBorderPreparationStatus = eqx.field(static=True)
    accepted: bool = eqx.field(static=True)
    required_border_edges: int = eqx.field(static=True)
    border_edge_capacity: int = eqx.field(static=True)
    required_quad_points: int = eqx.field(static=True)
    quad_point_capacity: int = eqx.field(static=True)
    incomplete_border_edge_count: int = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    resource_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class PlateauBorderPreparationError(ValueError):
    """The declared network support or resource capacity was refused."""

    def __init__(self, evidence: PlateauBorderPreparationEvidence, /) -> None:
        self.evidence = evidence
        super().__init__(
            f"Plateau-border preparation failed with status {evidence.status.name}."
        )


class PlateauBorderTransportError(ValueError):
    """A topology event has no declared physical transport of border content.

    Border liquid and surfactant cross an event pass only by an explicit rule:
    a retained border keeps its content and a split border shares it between
    its two children. T1 pops, pinches, merges, region splits, bursts, and
    border coarsening have no declared rule, so re-preparation is refused
    instead of redistributing content from lineage.
    """


@final
class PlateauBorderPlan(StrictModule, ParameterOwner):
    """Physical coefficients and fixed capacities of border drainage.

    ``hydraulic_shape_factor`` is the declared dimensionless resistance of the
    chosen triangular-channel closure.  It is not inferred or smoothed.
    ``capillary_shape_factor * sigma / sqrt(A)`` is the liquid-to-gas pressure
    magnitude.  Evaporation is disabled unless ``evaporation_declared`` is
    explicitly true; a nonzero sink on the disabled route is refused.
    """

    density_kg_m3: Array = parameter_field()
    viscosity_pa_s: Array = parameter_field()
    surface_tension_n_m: Array = parameter_field()
    gravity_m_s2: Array = parameter_field()
    hydraulic_shape_factor: Array = parameter_field()
    capillary_shape_factor: Array = parameter_field()
    time_step_s: Array = parameter_field()
    border_edge_capacity: int = eqx.field(static=True)
    quad_point_capacity: int = eqx.field(static=True)
    evaporation_declared: bool = eqx.field(static=True)
    maximum_courant_number: float = eqx.field(static=True)
    resource_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        border_edge_capacity: int,
        quad_point_capacity: int,
        density_kg_m3: ArrayLike,
        viscosity_pa_s: ArrayLike,
        surface_tension_n_m: ArrayLike,
        gravity_m_s2: ConvertibleToArray = (0.0, 0.0, -9.81),
        hydraulic_shape_factor: ArrayLike = 50.0,
        capillary_shape_factor: ArrayLike = 1.0,
        time_step_s: ArrayLike,
        evaporation_declared: bool = False,
        maximum_courant_number: float = 1.0,
        resource_id: str = "plateau-border-network",
    ) -> None:
        borders = positive_integer(border_edge_capacity, "border_edge_capacity")
        quads = nonnegative_integer(quad_point_capacity, "quad_point_capacity")
        density = _positive_scalar(density_kg_m3, "density_kg_m3")
        viscosity = _positive_scalar(viscosity_pa_s, "viscosity_pa_s")
        tension = _positive_scalar(surface_tension_n_m, "surface_tension_n_m")
        hydraulic = _positive_scalar(hydraulic_shape_factor, "hydraulic_shape_factor")
        capillary = _positive_scalar(capillary_shape_factor, "capillary_shape_factor")
        step_size = _positive_scalar(time_step_s, "time_step_s")
        gravity = np.asarray(gravity_m_s2, dtype=np.float64)
        if gravity.shape != (3,) or not np.all(np.isfinite(gravity)):
            raise ValueError("gravity_m_s2 must be one finite 3-vector.")
        if not isinstance(evaporation_declared, bool):
            raise TypeError("evaporation_declared must be a bool.")
        courant = float(maximum_courant_number)
        if not np.isfinite(courant) or courant <= 0.0 or courant > 1.0:
            raise ValueError("maximum_courant_number must lie in (0, 1].")
        resource = canonical_identifier(resource_id, "resource_id")
        self.density_kg_m3 = density
        self.viscosity_pa_s = viscosity
        self.surface_tension_n_m = tension
        self.gravity_m_s2 = jnp.asarray(gravity)
        self.hydraulic_shape_factor = hydraulic
        self.capillary_shape_factor = capillary
        self.time_step_s = step_size
        self.border_edge_capacity = borders
        self.quad_point_capacity = quads
        self.evaporation_declared = evaporation_declared
        self.maximum_courant_number = courant
        self.resource_id = resource
        self.plan_id = canonical_fingerprint(
            {
                "kind": "plateau-border-plan",
                "border_edge_capacity": borders,
                "quad_point_capacity": quads,
                "evaporation_declared": evaporation_declared,
                "maximum_courant_number": courant.hex(),
                "hydraulic_closure": "triangular-poiseuille-explicit-shape-factor",
                "capillary_closure": "inverse-square-root-area",
                "time_integration": "explicit-conservative-euler",
                "resource_id": resource,
            }
        )

    def prepare(
        self,
        surface: PreparedMultiRegionSurface,
        film_slots: PreparedFilmSheetSlots,
        state: MultiRegionSurfaceState,
        /,
        *,
        geometry_revision: ArrayLike = 0,
    ) -> PreparedPlateauBorder:
        """Prepare sparse network and sheet-boundary relations once."""
        return PreparedPlateauBorder(
            self,
            surface,
            film_slots,
            state,
            geometry_revision=geometry_revision,
        )


@final
class PlateauBorderBoundaryFlux(StrictModule):
    """Declared sheet exchange and evaporation rates over one border step.

    Sheet-to-border route rates are positive from the B film cell into the
    border.  Evaporation rates are nonnegative sinks and never carry
    surfactant implicitly.
    """

    sheet_to_border_liquid_m3_s: Array
    sheet_to_border_surfactant_mol_s: Array
    sheet_evaporation_m3_s: Array
    border_evaporation_m3_s: Array

    def __init__(
        self,
        sheet_to_border_liquid_m3_s: ArrayLike,
        sheet_to_border_surfactant_mol_s: ArrayLike,
        sheet_evaporation_m3_s: ArrayLike,
        border_evaporation_m3_s: ArrayLike,
        /,
    ) -> None:
        liquid = jnp.asarray(sheet_to_border_liquid_m3_s, dtype=jnp.float64)
        surfactant = jnp.asarray(sheet_to_border_surfactant_mol_s, dtype=jnp.float64)
        sheet_sink = jnp.asarray(sheet_evaporation_m3_s, dtype=jnp.float64)
        border_sink = jnp.asarray(border_evaporation_m3_s, dtype=jnp.float64)
        if liquid.ndim != 1 or surfactant.shape != liquid.shape:
            raise ValueError("Sheet-to-border liquid and surfactant must be vectors.")
        if sheet_sink.ndim != 2 or border_sink.ndim != 1:
            raise ValueError(
                "Sheet evaporation must be a matrix and border evaporation a vector."
            )
        self.sheet_to_border_liquid_m3_s = liquid
        self.sheet_to_border_surfactant_mol_s = surfactant
        self.sheet_evaporation_m3_s = sheet_sink
        self.border_evaporation_m3_s = border_sink


@final
class PlateauBorderState(StrictModule):
    """Extensive sheet and border state on one fixed topology epoch."""

    sheet_liquid_m3: Array
    sheet_surfactant_mol: Array
    border_liquid_m3: Array
    border_surfactant_mol: Array
    unresolved_rim_content_m3: Array
    geometry_revision: Array
    topology_id: str = eqx.field(static=True)

    def __init__(
        self,
        sheet_liquid_m3: ArrayLike,
        sheet_surfactant_mol: ArrayLike,
        border_liquid_m3: ArrayLike,
        border_surfactant_mol: ArrayLike,
        /,
        *,
        unresolved_rim_content_m3: ArrayLike = 0.0,
        geometry_revision: ArrayLike = 0,
        topology_id: str,
    ) -> None:
        if not isinstance(topology_id, str) or not topology_id:
            raise ValueError("topology_id must be a nonempty string.")
        sheet_liquid = jnp.asarray(sheet_liquid_m3, dtype=jnp.float64)
        sheet_surfactant = jnp.asarray(sheet_surfactant_mol, dtype=jnp.float64)
        border_liquid = jnp.asarray(border_liquid_m3, dtype=jnp.float64)
        border_surfactant = jnp.asarray(border_surfactant_mol, dtype=jnp.float64)
        rim = jnp.asarray(unresolved_rim_content_m3, dtype=jnp.float64)
        revision = jnp.asarray(geometry_revision, dtype=jnp.int32)
        if sheet_liquid.ndim != 2 or sheet_surfactant.shape != sheet_liquid.shape:
            raise ValueError("Sheet liquid and surfactant must be matching matrices.")
        if border_liquid.ndim != 1 or border_surfactant.shape != border_liquid.shape:
            raise ValueError("Border liquid and surfactant must be matching vectors.")
        if rim.shape != () or revision.shape != ():
            raise ValueError("Rim content and geometry revision must be scalars.")
        self.sheet_liquid_m3 = sheet_liquid
        self.sheet_surfactant_mol = sheet_surfactant
        self.border_liquid_m3 = border_liquid
        self.border_surfactant_mol = border_surfactant
        self.unresolved_rim_content_m3 = rim
        self.geometry_revision = revision
        self.topology_id = topology_id

    def total_liquid_m3(self) -> Array:
        """Film, border, and unresolved-rim liquid content."""
        return (
            jnp.sum(self.sheet_liquid_m3)
            + jnp.sum(self.border_liquid_m3)
            + self.unresolved_rim_content_m3
        )

    def total_surfactant_mol(self) -> Array:
        """Total represented sheet plus border surfactant amount."""
        return jnp.sum(self.sheet_surfactant_mol) + jnp.sum(self.border_surfactant_mol)


@final
class PlateauBorderRates(StrictModule):
    """Fixed-topology conservative rates and hydraulic diagnostics."""

    sheet_liquid_m3_s: Array
    sheet_surfactant_mol_s: Array
    border_liquid_m3_s: Array
    border_surfactant_mol_s: Array
    border_internal_flux_m3_s: Array
    border_internal_surfactant_flux_mol_s: Array
    cross_section_area_m2: Array
    border_pressure_pa: Array
    node_hydraulic_potential_pa: Array
    node_mass_balance_m3_s: Array
    node_pressure_balance_pa: Array
    border_outflow_courant_rate_s_inv: Array


@final
class PlateauBorderEvidence(StrictModule):
    """Conservation, quad balance, support, and acceptance evidence."""

    status: Array
    liquid_before_m3: Array
    liquid_after_m3: Array
    sheet_border_liquid_transfer_m3: Array
    liquid_evaporation_sink_m3: Array
    liquid_conservation_residual_m3: Array
    surfactant_before_mol: Array
    surfactant_after_mol: Array
    sheet_border_surfactant_transfer_mol: Array
    surfactant_conservation_residual_mol: Array
    maximum_quad_mass_residual_m3_s: Array
    maximum_quad_pressure_residual_pa: Array
    maximum_courant_number: Array
    minimum_cross_section_area_m2: Array
    unsupported_liquid_boundary_m3_s: Array
    unsupported_surfactant_boundary_mol_s: Array
    finite: Array
    positive: Array
    derivative_available: bool = eqx.field(static=True)
    evaporation_declared: bool = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @property
    def accepted(self) -> Array:
        return self.status == PlateauBorderStatus.ACCEPTED


@final
class PlateauBorderResult(StrictModule):
    """Committed or rolled-back border step plus its candidate and fluxes."""

    state: PlateauBorderState
    candidate_state: PlateauBorderState
    rates: PlateauBorderRates
    evidence: PlateauBorderEvidence

    @property
    def accepted(self) -> Array:
        return self.evidence.accepted


@final
class PlateauBorderRimEvidence(StrictModule):
    """Resolved versus unresolved E5 burst-liquid ledger."""

    status: PlateauBorderRimStatus = eqx.field(static=True)
    resolved: bool = eqx.field(static=True)
    resolved_content_m3: Array
    unresolved_content_m3: Array
    liquid_conservation_residual_m3: Array
    supporting_border_count: int = eqx.field(static=True)
    derivative_available: bool = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


@final
class PlateauBorderRimResult(StrictModule):
    """E5 rupture plus conservative rim allocation on the source epoch."""

    rupture: FoamRuptureResult
    border_state: PlateauBorderState
    border_content_added_m3: Array
    unresolved_rim_content_m3: Array
    evidence: PlateauBorderRimEvidence


@final
class PreparedPlateauBorder(StrictModule):
    """Fixed-capacity sparse Plateau-border network and B-on-E coupling."""

    plan: PlateauBorderPlan
    surface: PreparedMultiRegionSurface
    film_slots: PreparedFilmSheetSlots
    positions_m: Array
    border_global_edge_indices: Array
    border_edges: Array
    border_active: Array
    network_vertex_active: Array
    quad_vertex_active: Array
    endpoint_relation: EdgeRelation
    endpoint_to_borders: EdgeRelation
    boundary_to_borders: EdgeRelation
    boundary_supported: Array
    geometry_revision: Array
    preparation: PlateauBorderPreparationEvidence
    border_count: int = eqx.field(static=True)
    quad_point_count: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: PlateauBorderPlan,
        surface: PreparedMultiRegionSurface,
        film_slots: PreparedFilmSheetSlots,
        state: MultiRegionSurfaceState,
        /,
        *,
        geometry_revision: ArrayLike = 0,
    ) -> None:
        if not isinstance(plan, PlateauBorderPlan):
            raise TypeError("plan must be PlateauBorderPlan.")
        if not isinstance(surface, PreparedMultiRegionSurface):
            raise TypeError("surface must be PreparedMultiRegionSurface.")
        if not isinstance(film_slots, PreparedFilmSheetSlots):
            raise TypeError("film_slots must be PreparedFilmSheetSlots.")
        if not isinstance(state, MultiRegionSurfaceState):
            raise TypeError("state must be MultiRegionSurfaceState.")
        topology = surface.topology
        state.require_topology(topology)
        if film_slots.topology_id != topology.topology_id:
            self._refuse(
                plan,
                topology,
                PlateauBorderPreparationStatus.TOPOLOGY_MISMATCH,
                0,
                0,
                0,
            )
        revision = jnp.asarray(geometry_revision, dtype=jnp.int32)
        if revision.shape != ():
            raise ValueError("geometry_revision must be a scalar.")
        valence = np.sum(
            np.asarray(topology.edge_faces[: topology.edge_count]) >= 0, axis=1
        )
        physical = np.flatnonzero(valence == 3)
        border_count = physical.size
        edges = np.asarray(topology.edges[: topology.edge_count], dtype=np.int64)
        degree = np.bincount(
            edges[physical].reshape(-1), minlength=topology.vertex_capacity
        )
        quad = np.flatnonzero(degree == 4)
        route_edges = np.asarray(film_slots.boundary_global_edge_indices)
        route_supported = np.asarray(
            film_slots.boundary_route_valid & film_slots.boundary_plateau_supported
        )
        support_count = np.bincount(
            route_edges[route_supported], minlength=topology.edge_capacity
        )
        incomplete = int(np.count_nonzero(support_count[physical] != 6))
        if border_count == 0:
            self._refuse(
                plan,
                topology,
                PlateauBorderPreparationStatus.NO_PHYSICAL_BORDERS,
                border_count,
                quad.size,
                incomplete,
            )
        if border_count > plan.border_edge_capacity:
            self._refuse(
                plan,
                topology,
                PlateauBorderPreparationStatus.BORDER_CAPACITY_EXCEEDED,
                border_count,
                quad.size,
                incomplete,
            )
        if quad.size > plan.quad_point_capacity:
            self._refuse(
                plan,
                topology,
                PlateauBorderPreparationStatus.QUAD_POINT_CAPACITY_EXCEEDED,
                border_count,
                quad.size,
                incomplete,
            )
        if incomplete:
            self._refuse(
                plan,
                topology,
                PlateauBorderPreparationStatus.INCOMPLETE_SHEET_SUPPORT,
                border_count,
                quad.size,
                incomplete,
            )
        capacity = plan.border_edge_capacity
        border_global = np.zeros((capacity,), dtype=np.int64)
        border_global[:border_count] = physical
        border_edges = np.zeros((capacity, 2), dtype=np.int64)
        border_edges[:border_count] = edges[physical]
        active = np.arange(capacity) < border_count
        endpoint_count = 2 * capacity
        endpoint_border = np.repeat(np.arange(capacity), 2)
        endpoint_vertex = border_edges.reshape(-1)
        endpoint_valid = np.repeat(active, 2)
        local_of_global = np.full((topology.edge_capacity,), -1, dtype=np.int64)
        local_of_global[physical] = np.arange(border_count)
        boundary_local = local_of_global[route_edges]
        boundary_valid = route_supported & (boundary_local >= 0)
        safe_boundary_local = np.maximum(boundary_local, 0)
        network_active = degree > 0
        quad_active = degree == 4
        preparation = _preparation_evidence(
            plan,
            topology,
            PlateauBorderPreparationStatus.ACCEPTED,
            border_count,
            quad.size,
            incomplete,
        )
        self.plan = plan
        self.surface = surface
        self.film_slots = film_slots
        self.positions_m = jnp.asarray(state.positions, dtype=jnp.float64)
        self.border_global_edge_indices = jnp.asarray(border_global, dtype=jnp.int32)
        self.border_edges = jnp.asarray(border_edges, dtype=jnp.int32)
        self.border_active = jnp.asarray(active)
        self.network_vertex_active = jnp.asarray(network_active)
        self.quad_vertex_active = jnp.asarray(quad_active)
        self.endpoint_relation = EdgeRelation(
            endpoint_border,
            endpoint_vertex,
            source_size=capacity,
            target_size=topology.vertex_capacity,
            valid=endpoint_valid,
        )
        self.endpoint_to_borders = EdgeRelation(
            np.arange(endpoint_count),
            endpoint_border,
            source_size=endpoint_count,
            target_size=capacity,
            valid=endpoint_valid,
        )
        self.boundary_to_borders = EdgeRelation(
            np.arange(film_slots.boundary_route_capacity),
            safe_boundary_local,
            source_size=film_slots.boundary_route_capacity,
            target_size=capacity,
            valid=boundary_valid,
        )
        self.boundary_supported = jnp.asarray(boundary_valid)
        self.geometry_revision = revision
        self.preparation = preparation
        self.border_count = border_count
        self.quad_point_count = quad.size
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-plateau-border",
                "plan": plan.plan_id,
                "topology": topology.topology_id,
                "lineage": topology.lineage_id,
                "film_adapter": film_slots.adapter_id,
                "border_edges": array_tree_fingerprint(physical),
                "quad_vertices": array_tree_fingerprint(quad),
            }
        )

    @staticmethod
    def _refuse(
        plan: PlateauBorderPlan,
        topology: MultiRegionSurfaceTopology,
        status: PlateauBorderPreparationStatus,
        border_count: int,
        quad_count: int,
        incomplete: int,
    ) -> NoReturn:
        raise PlateauBorderPreparationError(
            _preparation_evidence(
                plan, topology, status, border_count, quad_count, incomplete
            )
        )

    def initial_state(
        self,
        sheet_liquid_m3: ArrayLike,
        sheet_surfactant_mol: ArrayLike,
        cross_section_area_m2: ArrayLike,
        /,
        *,
        border_surfactant_concentration_mol_m3: ArrayLike = 0.0,
        unresolved_rim_content_m3: ArrayLike = 0.0,
    ) -> PlateauBorderState:
        """Build extensive border content from cross-section and concentration."""
        area = jnp.broadcast_to(
            jnp.asarray(cross_section_area_m2, dtype=jnp.float64),
            (self.plan.border_edge_capacity,),
        )
        concentration = jnp.broadcast_to(
            jnp.asarray(border_surfactant_concentration_mol_m3, dtype=jnp.float64),
            (self.plan.border_edge_capacity,),
        )
        length = self._geometry()[0]
        volume = jnp.where(self.border_active, area * length, 0.0)
        return PlateauBorderState(
            sheet_liquid_m3,
            sheet_surfactant_mol,
            volume,
            concentration * volume,
            unresolved_rim_content_m3=unresolved_rim_content_m3,
            geometry_revision=self.geometry_revision,
            topology_id=self.surface.topology.topology_id,
        )

    def zero_boundary_flux(self) -> PlateauBorderBoundaryFlux:
        """Return an explicit closed, non-evaporating boundary declaration."""
        topology = self.surface.topology
        return PlateauBorderBoundaryFlux(
            jnp.zeros((self.film_slots.boundary_route_capacity,)),
            jnp.zeros((self.film_slots.boundary_route_capacity,)),
            jnp.zeros((topology.vertex_capacity, topology.slot_width)),
            jnp.zeros((self.plan.border_edge_capacity,)),
        )

    def rates(
        self,
        state: PlateauBorderState,
        boundary: PlateauBorderBoundaryFlux,
        /,
    ) -> PlateauBorderRates:
        """Return conservative fixed-topology rates without committing a step."""
        self._check_shapes(state, boundary)
        length, midpoint = self._geometry()
        safe_volume = jnp.where(self.border_active, state.border_liquid_m3, 1.0)
        area = jnp.where(self.border_active, safe_volume / length, 0.0)
        safe_area = jnp.where(self.border_active, area, 1.0)
        pressure = jnp.where(
            self.border_active,
            -self.plan.capillary_shape_factor
            * self.plan.surface_tension_n_m
            / jnp.sqrt(safe_area),
            0.0,
        )
        potential = pressure - self.plan.density_kg_m3 * (
            midpoint @ self.plan.gravity_m_s2
        )
        conductance = jnp.where(
            self.border_active,
            2.0
            * safe_area**2
            / (self.plan.hydraulic_shape_factor * self.plan.viscosity_pa_s * length),
            0.0,
        )
        route_conductance = gather_routes(self.endpoint_relation, conductance)
        route_potential = gather_routes(self.endpoint_relation, potential)
        node_conductance = route_reduce(self.endpoint_relation, route_conductance)
        node_weighted = route_reduce(
            self.endpoint_relation, route_conductance * route_potential
        )
        safe_node_conductance = jnp.where(node_conductance > 0.0, node_conductance, 1.0)
        node_potential = jnp.where(
            self.network_vertex_active,
            node_weighted / safe_node_conductance,
            0.0,
        )
        endpoint_vertices = self.endpoint_relation.target_indices
        hydraulic_flux = route_conductance * (
            route_potential - node_potential[endpoint_vertices]
        )
        hydraulic_flux = jnp.where(self.endpoint_relation.valid, hydraulic_flux, 0.0)
        node_mass = route_reduce(self.endpoint_relation, hydraulic_flux)
        node_pressure = node_mass / safe_node_conductance
        border_outflow = route_reduce(self.endpoint_to_borders, hydraulic_flux)
        internal_liquid_rate = -border_outflow

        concentration = jnp.where(
            self.border_active,
            state.border_surfactant_mol / safe_volume,
            0.0,
        )
        route_concentration = gather_routes(self.endpoint_relation, concentration)
        incoming_amount = route_reduce(
            self.endpoint_relation,
            jnp.maximum(hydraulic_flux, 0.0) * route_concentration,
        )
        node_liquid_outflow = route_reduce(
            self.endpoint_relation, jnp.maximum(-hydraulic_flux, 0.0)
        )
        node_concentration = incoming_amount / jnp.where(
            node_liquid_outflow > 0.0, node_liquid_outflow, 1.0
        )
        internal_surfactant_flux = jnp.where(
            hydraulic_flux >= 0.0,
            hydraulic_flux * route_concentration,
            hydraulic_flux * node_concentration[endpoint_vertices],
        )
        internal_surfactant_rate = -route_reduce(
            self.endpoint_to_borders, internal_surfactant_flux
        )

        liquid_boundary = jnp.where(
            self.boundary_supported,
            boundary.sheet_to_border_liquid_m3_s,
            0.0,
        )
        surfactant_boundary = jnp.where(
            self.boundary_supported,
            boundary.sheet_to_border_surfactant_mol_s,
            0.0,
        )
        sheet_liquid_rate = -self.film_slots.boundary_slot_rate(liquid_boundary)
        sheet_surfactant_rate = -self.film_slots.boundary_slot_rate(surfactant_boundary)
        border_liquid_boundary = route_reduce(self.boundary_to_borders, liquid_boundary)
        border_surfactant_boundary = route_reduce(
            self.boundary_to_borders, surfactant_boundary
        )
        sheet_evaporation = jnp.where(
            self.surface.topology.slot_active,
            boundary.sheet_evaporation_m3_s,
            0.0,
        )
        border_evaporation = jnp.where(
            self.border_active, boundary.border_evaporation_m3_s, 0.0
        )
        sheet_liquid_rate = sheet_liquid_rate - sheet_evaporation
        border_liquid_rate = (
            internal_liquid_rate + border_liquid_boundary - border_evaporation
        )
        border_surfactant_rate = internal_surfactant_rate + border_surfactant_boundary
        endpoint_outflow = route_reduce(
            self.endpoint_to_borders, jnp.maximum(hydraulic_flux, 0.0)
        )
        boundary_outflow = route_reduce(
            self.boundary_to_borders, jnp.maximum(-liquid_boundary, 0.0)
        )
        outflow_rate = jnp.where(
            self.border_active,
            (endpoint_outflow + boundary_outflow + border_evaporation) / safe_volume,
            0.0,
        )
        return PlateauBorderRates(
            sheet_liquid_m3_s=sheet_liquid_rate,
            sheet_surfactant_mol_s=sheet_surfactant_rate,
            border_liquid_m3_s=border_liquid_rate,
            border_surfactant_mol_s=border_surfactant_rate,
            border_internal_flux_m3_s=hydraulic_flux,
            border_internal_surfactant_flux_mol_s=internal_surfactant_flux,
            cross_section_area_m2=area,
            border_pressure_pa=pressure,
            node_hydraulic_potential_pa=node_potential,
            node_mass_balance_m3_s=node_mass,
            node_pressure_balance_pa=node_pressure,
            border_outflow_courant_rate_s_inv=outflow_rate,
        )

    def step(
        self,
        state: PlateauBorderState,
        boundary: PlateauBorderBoundaryFlux,
        /,
    ) -> PlateauBorderResult:
        """Advance one conservative explicit border step or roll back unchanged."""
        rates = self.rates(state, boundary)
        dt = self.plan.time_step_s
        candidate = PlateauBorderState(
            state.sheet_liquid_m3 + dt * rates.sheet_liquid_m3_s,
            state.sheet_surfactant_mol + dt * rates.sheet_surfactant_mol_s,
            state.border_liquid_m3 + dt * rates.border_liquid_m3_s,
            state.border_surfactant_mol + dt * rates.border_surfactant_mol_s,
            unresolved_rim_content_m3=state.unresolved_rim_content_m3,
            geometry_revision=state.geometry_revision,
            topology_id=state.topology_id,
        )
        topology = self.surface.topology
        slot_active = topology.slot_active
        active_border = self.border_active
        boundary_finite = (
            jnp.all(jnp.isfinite(boundary.sheet_to_border_liquid_m3_s))
            & jnp.all(jnp.isfinite(boundary.sheet_to_border_surfactant_mol_s))
            & jnp.all(jnp.isfinite(boundary.sheet_evaporation_m3_s))
            & jnp.all(jnp.isfinite(boundary.border_evaporation_m3_s))
        )
        evaporation_nonnegative = jnp.all(
            boundary.sheet_evaporation_m3_s >= 0.0
        ) & jnp.all(boundary.border_evaporation_m3_s >= 0.0)
        padding_zero = (
            jnp.all(
                jnp.where(
                    self.film_slots.boundary_route_valid,
                    True,
                    (boundary.sheet_to_border_liquid_m3_s == 0.0)
                    & (boundary.sheet_to_border_surfactant_mol_s == 0.0),
                )
            )
            & jnp.all(
                jnp.where(slot_active, True, boundary.sheet_evaporation_m3_s == 0.0)
            )
            & jnp.all(
                jnp.where(active_border, True, boundary.border_evaporation_m3_s == 0.0)
            )
        )
        state_finite = (
            jnp.all(jnp.isfinite(state.sheet_liquid_m3))
            & jnp.all(jnp.isfinite(state.sheet_surfactant_mol))
            & jnp.all(jnp.isfinite(state.border_liquid_m3))
            & jnp.all(jnp.isfinite(state.border_surfactant_mol))
            & jnp.isfinite(state.unresolved_rim_content_m3)
        )
        admissible = (
            state_finite
            & boundary_finite
            & evaporation_nonnegative
            & padding_zero
            & jnp.all(jnp.where(slot_active, state.sheet_liquid_m3 >= 0.0, True))
            & jnp.all(jnp.where(slot_active, state.sheet_surfactant_mol >= 0.0, True))
            & jnp.all(jnp.where(slot_active, True, state.sheet_liquid_m3 == 0.0))
            & jnp.all(jnp.where(slot_active, True, state.sheet_surfactant_mol == 0.0))
            & jnp.all(jnp.where(active_border, state.border_liquid_m3 > 0.0, True))
            & jnp.all(jnp.where(active_border, state.border_surfactant_mol >= 0.0, True))
            & jnp.all(jnp.where(active_border, True, state.border_liquid_m3 == 0.0))
            & jnp.all(jnp.where(active_border, True, state.border_surfactant_mol == 0.0))
            & (state.unresolved_rim_content_m3 >= 0.0)
        )
        revision_matches = state.geometry_revision == self.geometry_revision
        unsupported_liquid = self.film_slots.unsupported_boundary_rate(
            boundary.sheet_to_border_liquid_m3_s
        )
        unsupported_surfactant = self.film_slots.unsupported_boundary_rate(
            boundary.sheet_to_border_surfactant_mol_s
        )
        evaporation = jnp.sum(
            jnp.where(slot_active, boundary.sheet_evaporation_m3_s, 0.0)
        ) + jnp.sum(jnp.where(active_border, boundary.border_evaporation_m3_s, 0.0))
        evaporation_declared = self.plan.evaporation_declared | (evaporation == 0.0)
        maximum_courant = dt * jnp.max(rates.border_outflow_courant_rate_s_inv)
        positive = (
            jnp.all(jnp.where(slot_active, candidate.sheet_liquid_m3 >= 0.0, True))
            & jnp.all(jnp.where(slot_active, candidate.sheet_surfactant_mol >= 0.0, True))
            & jnp.all(jnp.where(active_border, candidate.border_liquid_m3 > 0.0, True))
            & jnp.all(
                jnp.where(active_border, candidate.border_surfactant_mol >= 0.0, True)
            )
        )
        finite = (
            jnp.all(jnp.isfinite(candidate.sheet_liquid_m3))
            & jnp.all(jnp.isfinite(candidate.sheet_surfactant_mol))
            & jnp.all(jnp.isfinite(candidate.border_liquid_m3))
            & jnp.all(jnp.isfinite(candidate.border_surfactant_mol))
            & jnp.all(jnp.isfinite(rates.border_pressure_pa))
        )
        status = _resolve_status(
            (PlateauBorderStatus.INADMISSIBLE_INPUT, ~admissible),
            (
                PlateauBorderStatus.GEOMETRY_REVISION_MISMATCH,
                ~revision_matches,
            ),
            (
                PlateauBorderStatus.UNSUPPORTED_BOUNDARY_FLUX,
                (unsupported_liquid > 0.0) | (unsupported_surfactant > 0.0),
            ),
            (PlateauBorderStatus.EVAPORATION_UNDECLARED, ~evaporation_declared),
            (
                PlateauBorderStatus.COURANT_LIMIT,
                maximum_courant > self.plan.maximum_courant_number,
            ),
            (PlateauBorderStatus.NEGATIVE_CONTENT, ~positive),
            (PlateauBorderStatus.NONFINITE, ~finite),
        )
        accepted = status == PlateauBorderStatus.ACCEPTED
        committed = PlateauBorderState(
            jnp.where(accepted, candidate.sheet_liquid_m3, state.sheet_liquid_m3),
            jnp.where(
                accepted,
                candidate.sheet_surfactant_mol,
                state.sheet_surfactant_mol,
            ),
            jnp.where(accepted, candidate.border_liquid_m3, state.border_liquid_m3),
            jnp.where(
                accepted,
                candidate.border_surfactant_mol,
                state.border_surfactant_mol,
            ),
            unresolved_rim_content_m3=state.unresolved_rim_content_m3,
            geometry_revision=state.geometry_revision,
            topology_id=state.topology_id,
        )
        liquid_before = state.total_liquid_m3()
        liquid_after = candidate.total_liquid_m3()
        surfactant_before = state.total_surfactant_mol()
        surfactant_after = candidate.total_surfactant_mol()
        liquid_transfer = dt * jnp.sum(
            jnp.where(
                self.boundary_supported,
                boundary.sheet_to_border_liquid_m3_s,
                0.0,
            )
        )
        surfactant_transfer = dt * jnp.sum(
            jnp.where(
                self.boundary_supported,
                boundary.sheet_to_border_surfactant_mol_s,
                0.0,
            )
        )
        evaporation_sink = dt * evaporation
        quad_mass = jnp.max(
            jnp.where(self.quad_vertex_active, jnp.abs(rates.node_mass_balance_m3_s), 0.0)
        )
        quad_pressure = jnp.max(
            jnp.where(
                self.quad_vertex_active,
                jnp.abs(rates.node_pressure_balance_pa),
                0.0,
            )
        )
        minimum_area = jnp.min(
            jnp.where(active_border, rates.cross_section_area_m2, jnp.inf)
        )
        evidence = PlateauBorderEvidence(
            status=jnp.asarray(status, dtype=jnp.int32),
            liquid_before_m3=liquid_before,
            liquid_after_m3=liquid_after,
            sheet_border_liquid_transfer_m3=liquid_transfer,
            liquid_evaporation_sink_m3=evaporation_sink,
            liquid_conservation_residual_m3=liquid_after
            + evaporation_sink
            - liquid_before,
            surfactant_before_mol=surfactant_before,
            surfactant_after_mol=surfactant_after,
            sheet_border_surfactant_transfer_mol=surfactant_transfer,
            surfactant_conservation_residual_mol=surfactant_after - surfactant_before,
            maximum_quad_mass_residual_m3_s=quad_mass,
            maximum_quad_pressure_residual_pa=quad_pressure,
            maximum_courant_number=maximum_courant,
            minimum_cross_section_area_m2=minimum_area,
            unsupported_liquid_boundary_m3_s=unsupported_liquid,
            unsupported_surfactant_boundary_mol_s=unsupported_surfactant,
            finite=finite,
            positive=positive,
            derivative_available=True,
            evaporation_declared=self.plan.evaporation_declared,
            topology_id=self.surface.topology.topology_id,
            prepared_id=self.prepared_id,
            plan_id=self.plan.plan_id,
        )
        return PlateauBorderResult(committed, candidate, rates, evidence)

    def refresh(
        self,
        surface: PreparedMultiRegionSurface,
        film_slots: PreparedFilmSheetSlots,
        state: MultiRegionSurfaceState,
        /,
        *,
        geometry_revision: ArrayLike,
    ) -> PreparedPlateauBorder:
        """Refresh numeric B4 geometry while preserving sparse network structure."""
        if surface.topology.topology_id != self.surface.topology.topology_id:
            raise ValueError("A Plateau-border refresh must preserve topology.")
        if film_slots.adapter_id != self.film_slots.adapter_id:
            raise ValueError("A Plateau-border refresh must preserve sheet-slot maps.")
        state.require_topology(surface.topology)
        revision = jnp.asarray(geometry_revision, dtype=jnp.int32)
        if revision.shape != ():
            raise ValueError("geometry_revision must be a scalar.")
        return eqx.tree_at(
            lambda value: (
                value.surface,
                value.film_slots,
                value.positions_m,
                value.geometry_revision,
            ),
            self,
            (
                surface,
                film_slots,
                jnp.asarray(state.positions, dtype=jnp.float64),
                revision,
            ),
        )

    def reprepare_after_events(
        self,
        event: SurfaceEventPassEvidence,
        surface: PreparedMultiRegionSurface,
        film_slots: PreparedFilmSheetSlots,
        state: MultiRegionSurfaceState,
        /,
        *,
        geometry_revision: ArrayLike = 0,
    ) -> PlateauBorderAdaptation:
        """Reprepare the network on a committed event target and transport content.

        Host boundary without derivatives. ``event`` is the committed pass
        from this network's epoch and geometry; ``surface``, ``film_slots``,
        and ``state`` are prepared on its target geometry. An accepted event
        without a declared border-content rule raises
        :class:`PlateauBorderTransportError` before any target network is
        prepared, so this network and its state stay usable.
        """
        lineage, target_epoch = _committed_event(self, event, surface, state)
        _require_border_transport_rule(event)
        target = PreparedPlateauBorder(
            self.plan, surface, film_slots, state, geometry_revision=geometry_revision
        )
        sources, targets, weights = _border_event_routes(self, target, lineage)
        transfer = ConservativeFieldTransfer(
            sources,
            targets,
            weights,
            source_active=np.asarray(self.border_active),
            target_active=np.asarray(target.border_active),
        )
        transition = transfer.epoch_transition(
            event.source_epoch, target_epoch, field_name="plateau-border-content"
        )
        fan_out = np.bincount(sources, minlength=self.border_count)
        return PlateauBorderAdaptation(
            target,
            transfer,
            transition,
            retained_border_count=int(np.count_nonzero(fan_out == 1)),
            refined_border_count=int(np.count_nonzero(fan_out == 2)),
        )

    def _rim_weights(self, region_pair: tuple[str, str], /) -> tuple[Array, int]:
        canonical = tuple(sorted(region_pair))
        sheet_indices = tuple(
            index
            for index, view in enumerate(self.film_slots.views.views)
            if view.region_ids == canonical
        )
        if len(sheet_indices) != 1:
            return jnp.zeros_like(self.border_active, dtype=jnp.float64), 0
        selected = self.boundary_supported & (
            self.film_slots.boundary_sheet_indices == sheet_indices[0]
        )
        edge_selected = route_reduce(
            self.boundary_to_borders, selected.astype(jnp.float64)
        )
        support = edge_selected > 0.0
        count = int(np.count_nonzero(np.asarray(support)))
        if count == 0:
            return jnp.zeros_like(self.border_active, dtype=jnp.float64), 0
        length, _ = self._geometry()
        weights = jnp.where(support, length, 0.0)
        return weights / jnp.sum(weights), count

    def resolve_rim_content(
        self,
        state: PlateauBorderState,
        region_pair: tuple[str, str],
        content_m3: ArrayLike,
        /,
    ) -> tuple[PlateauBorderState, Array, int]:
        """Allocate burst liquid to the physical border bounding one sheet.

        This event-boundary operation returns a source-epoch border transfer
        record.  The caller must reprepare after the topology transition; no
        derivative is claimed across it.
        """
        self._check_state_shape(state)
        weights, count = self._rim_weights(region_pair)
        if count == 0:
            return state, jnp.zeros_like(state.border_liquid_m3), 0
        amount = jnp.asarray(content_m3, dtype=jnp.float64)
        if amount.shape != ():
            raise ValueError("content_m3 must be a scalar.")
        added = amount * weights
        resolved = PlateauBorderState(
            state.sheet_liquid_m3,
            state.sheet_surfactant_mol,
            state.border_liquid_m3 + added,
            state.border_surfactant_mol,
            unresolved_rim_content_m3=state.unresolved_rim_content_m3,
            geometry_revision=state.geometry_revision,
            topology_id=state.topology_id,
        )
        return resolved, added, count

    def _geometry(self) -> tuple[Array, Array]:
        first = self.positions_m[self.border_edges[:, 0]]
        second = self.positions_m[self.border_edges[:, 1]]
        length = jnp.linalg.norm(second - first, axis=1)
        length = jnp.where(self.border_active, length, 1.0)
        return length, 0.5 * (first + second)

    def _check_state_shape(self, state: PlateauBorderState) -> None:
        if not isinstance(state, PlateauBorderState):
            raise TypeError("state must be PlateauBorderState.")
        topology = self.surface.topology
        slot_shape = (topology.vertex_capacity, topology.slot_width)
        border_shape = (self.plan.border_edge_capacity,)
        if state.topology_id != topology.topology_id:
            raise ValueError("State topology does not match the prepared network.")
        if (
            state.sheet_liquid_m3.shape != slot_shape
            or state.sheet_surfactant_mol.shape != slot_shape
        ):
            raise ValueError("Sheet content must use the E sheet-slot layout.")
        if (
            state.border_liquid_m3.shape != border_shape
            or state.border_surfactant_mol.shape != border_shape
        ):
            raise ValueError("Border content must use border_edge_capacity.")
        if state.unresolved_rim_content_m3.shape != ():
            raise ValueError("unresolved_rim_content_m3 must be a scalar.")
        if state.geometry_revision.shape != ():
            raise ValueError("geometry_revision must be a scalar.")

    def _check_shapes(
        self, state: PlateauBorderState, boundary: PlateauBorderBoundaryFlux, /
    ) -> None:
        self._check_state_shape(state)
        if not isinstance(boundary, PlateauBorderBoundaryFlux):
            raise TypeError("boundary must be PlateauBorderBoundaryFlux.")
        topology = self.surface.topology
        if boundary.sheet_to_border_liquid_m3_s.shape != (
            self.film_slots.boundary_route_capacity,
        ) or boundary.sheet_to_border_surfactant_mol_s.shape != (
            self.film_slots.boundary_route_capacity,
        ):
            raise ValueError("Sheet-to-border rates must use boundary_route_capacity.")
        if boundary.sheet_evaporation_m3_s.shape != (
            topology.vertex_capacity,
            topology.slot_width,
        ):
            raise ValueError("Sheet evaporation must use the E sheet-slot layout.")
        if boundary.border_evaporation_m3_s.shape != (self.plan.border_edge_capacity,):
            raise ValueError("Border evaporation must use border_edge_capacity.")


@final
class PlateauBorderAdaptation(StrictModule):
    """Target-epoch border network plus the explicit transport of its content.

    ``transition`` is the nondifferentiable epoch transition of one extensive
    border field (liquid volume or surfactant amount) from the source to the
    target border axis. A retained border keeps its content. A border split
    at a new vertex gives each child the fraction ``L_child / (L_1 + L_2)`` of
    its content, which keeps the cell's uniform cross-section ``V / L`` and
    surfactant concentration. The weights of every source border sum to one,
    so totals are conserved to roundoff and content stays nonnegative.
    """

    prepared: PreparedPlateauBorder
    transfer: ConservativeFieldTransfer
    transition: TopologyEpochTransition
    retained_border_count: int = eqx.field(static=True)
    refined_border_count: int = eqx.field(static=True)
    adaptation_id: str = eqx.field(static=True)

    def __init__(
        self,
        prepared: PreparedPlateauBorder,
        transfer: ConservativeFieldTransfer,
        transition: TopologyEpochTransition,
        /,
        *,
        retained_border_count: int,
        refined_border_count: int,
    ) -> None:
        if not isinstance(prepared, PreparedPlateauBorder):
            raise TypeError("prepared must be PreparedPlateauBorder.")
        if not isinstance(transfer, ConservativeFieldTransfer):
            raise TypeError("transfer must be ConservativeFieldTransfer.")
        if not isinstance(transition, TopologyEpochTransition):
            raise TypeError("transition must be TopologyEpochTransition.")
        retained = nonnegative_integer(retained_border_count, "retained_border_count")
        refined = nonnegative_integer(refined_border_count, "refined_border_count")
        self.prepared = prepared
        self.transfer = transfer
        self.transition = transition
        self.retained_border_count = retained
        self.refined_border_count = refined
        self.adaptation_id = canonical_fingerprint(
            {
                "kind": "plateau-border-adaptation",
                "target": prepared.prepared_id,
                "transition": transition.transition_id,
                "retained": retained,
                "refined": refined,
            }
        )


def apply_foam_rupture_with_borders(
    rupture_plan: FoamRupturePlan,
    prepared: PreparedPlateauBorder,
    border_state: PlateauBorderState,
    topology: MultiRegionSurfaceTopology,
    surface_state: MultiRegionSurfaceState,
    thickness_m: ArrayLike,
    film_status: ArrayLike,
    film_evidence: SurfaceFilmEvidence,
    geometry_revision: ArrayLike,
    /,
    *,
    event_policy: SurfaceEventPolicy | None = None,
) -> PlateauBorderRimResult:
    """Apply public E5 rupture and resolve its new rim ledger when supported.

    Unsupported or rolled-back events keep E5's unresolved ledger exactly;
    content is never dropped.  A resolved allocation is a source-epoch event
    transfer and must be followed by normal host re-preparation for the target
    topology epoch.
    """
    if not isinstance(prepared, PreparedPlateauBorder):
        raise TypeError("prepared must be PreparedPlateauBorder.")
    if not isinstance(film_evidence, SurfaceFilmEvidence):
        raise TypeError("film_evidence must be SurfaceFilmEvidence.")
    if topology.topology_id != prepared.surface.topology.topology_id:
        raise ValueError("Rupture and Plateau-border topology must match.")
    prepared._check_state_shape(border_state)
    if rupture_plan.liquid_field_name in surface_state.sheet_field_names:
        liquid_index = surface_state.sheet_field_names.index(
            rupture_plan.liquid_field_name
        )
        if not np.array_equal(
            np.asarray(border_state.sheet_liquid_m3),
            np.asarray(surface_state.sheet_fields[..., liquid_index]),
        ):
            raise ValueError(
                "Plateau-border and multiregion film liquid must share one "
                "sheet-slot state before rupture."
            )
    rupture = apply_foam_rupture(
        rupture_plan,
        topology,
        surface_state,
        thickness_m,
        film_status,
        film_evidence,
        geometry_revision,
        border_state.unresolved_rim_content_m3,
        event_policy=event_policy,
    )
    if not rupture.successful or rupture.proposal is None:
        evidence = PlateauBorderRimEvidence(
            status=PlateauBorderRimStatus.RUPTURE_NOT_COMMITTED,
            resolved=False,
            resolved_content_m3=jnp.asarray(0.0),
            unresolved_content_m3=rupture.unresolved_rim_content,
            liquid_conservation_residual_m3=jnp.asarray(0.0),
            supporting_border_count=0,
            derivative_available=False,
            prepared_id=prepared.prepared_id,
        )
        return PlateauBorderRimResult(
            rupture,
            border_state,
            jnp.zeros_like(border_state.border_liquid_m3),
            rupture.unresolved_rim_content,
            evidence,
        )
    weights, support_count = prepared._rim_weights(rupture.proposal.region_ids)
    if support_count == 0:
        evidence = PlateauBorderRimEvidence(
            status=PlateauBorderRimStatus.BORDER_SUPPORT_UNAVAILABLE,
            resolved=False,
            resolved_content_m3=jnp.asarray(
                0.0, dtype=rupture.unresolved_rim_content.dtype
            ),
            unresolved_content_m3=rupture.unresolved_rim_content,
            liquid_conservation_residual_m3=jnp.asarray(
                0.0, dtype=rupture.unresolved_rim_content.dtype
            ),
            supporting_border_count=0,
            derivative_available=False,
            prepared_id=prepared.prepared_id,
        )
        return PlateauBorderRimResult(
            rupture,
            border_state,
            jnp.zeros_like(border_state.border_liquid_m3),
            rupture.unresolved_rim_content,
            evidence,
        )
    _, dropped = _rupture_sheet_masks(topology, rupture)
    dropped_mask = jnp.asarray(dropped)
    dropped_surfactant = jnp.sum(
        jnp.where(dropped_mask, border_state.sheet_surfactant_mol, 0.0)
    )
    amount = rupture.evidence.unresolved_rim_added
    added = amount * weights
    resolved_state = PlateauBorderState(
        jnp.where(dropped_mask, 0.0, border_state.sheet_liquid_m3),
        jnp.where(dropped_mask, 0.0, border_state.sheet_surfactant_mol),
        border_state.border_liquid_m3 + added,
        border_state.border_surfactant_mol + dropped_surfactant * weights,
        unresolved_rim_content_m3=rupture.evidence.unresolved_rim_before,
        geometry_revision=border_state.geometry_revision,
        topology_id=border_state.topology_id,
    )
    resolved_content = jnp.sum(added)
    unresolved = rupture.evidence.unresolved_rim_before
    residual = resolved_state.total_liquid_m3() - border_state.total_liquid_m3()
    evidence = PlateauBorderRimEvidence(
        status=PlateauBorderRimStatus.RESOLVED,
        resolved=True,
        resolved_content_m3=resolved_content,
        unresolved_content_m3=unresolved,
        liquid_conservation_residual_m3=residual,
        supporting_border_count=support_count,
        derivative_available=False,
        prepared_id=prepared.prepared_id,
    )
    return PlateauBorderRimResult(rupture, resolved_state, added, unresolved, evidence)


def _rupture_sheet_masks(
    source: MultiRegionSurfaceTopology,
    rupture: FoamRuptureResult,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Identify source slots transferred to the target and dropped into the rim."""
    if rupture.proposal is None:
        raise ValueError("A committed rupture must carry its proposal.")
    target = rupture.topology
    source_pair = rupture.proposal.region_ids
    surviving_ids = tuple(
        region_id for region_id in source_pair if region_id in target.region_ids
    )
    removed_ids = tuple(
        region_id for region_id in source_pair if region_id not in target.region_ids
    )
    if len(surviving_ids) != 1 or len(removed_ids) != 1:
        raise ValueError("A committed rupture must merge exactly one source region.")
    survivor_id = surviving_ids[0]
    removed_id = removed_ids[0]
    target_lookup: set[tuple[int, tuple[str, str]]] = set()
    target_vertex_ids = np.asarray(
        target.vertex_global_ids[: target.vertex_count], dtype=np.int64
    )
    target_pairs = np.asarray(
        target.region_pairs[: target.region_pair_count], dtype=np.int64
    )
    target_slots = np.asarray(
        target.vertex_pair_slots[: target.vertex_count], dtype=np.int64
    )
    for vertex in range(target.vertex_count):
        for slot in range(target.slot_width):
            pair_index = target_slots[vertex, slot]
            if pair_index < 0:
                continue
            labels = target_pairs[pair_index]
            pair_ids = tuple(
                sorted((target.region_ids[labels[0]], target.region_ids[labels[1]]))
            )
            target_lookup.add((int(target_vertex_ids[vertex]), pair_ids))

    survivor = np.zeros((source.vertex_capacity, source.slot_width), dtype=np.bool_)
    source_vertex_ids = np.asarray(
        source.vertex_global_ids[: source.vertex_count], dtype=np.int64
    )
    source_pairs = np.asarray(
        source.region_pairs[: source.region_pair_count], dtype=np.int64
    )
    source_slots = np.asarray(
        source.vertex_pair_slots[: source.vertex_count], dtype=np.int64
    )
    for vertex in range(source.vertex_count):
        for slot in range(source.slot_width):
            pair_index = source_slots[vertex, slot]
            if pair_index < 0:
                continue
            labels = source_pairs[pair_index]
            mapped = tuple(
                sorted(
                    survivor_id
                    if source.region_ids[label] == removed_id
                    else source.region_ids[label]
                    for label in labels
                )
            )
            survivor[vertex, slot] = (
                mapped[0] != mapped[1]
                and (int(source_vertex_ids[vertex]), mapped) in target_lookup
            )
    active = np.asarray(source.slot_active, dtype=np.bool_)
    return survivor, active & ~survivor


def _committed_event(
    source: PreparedPlateauBorder,
    event: SurfaceEventPassEvidence,
    surface: PreparedMultiRegionSurface,
    state: MultiRegionSurfaceState,
    /,
) -> tuple[MultiRegionSurfaceLineage, TopologyEpoch]:
    """Lineage and target epoch of a committed pass from ``source``'s geometry."""
    if not isinstance(event, SurfaceEventPassEvidence):
        raise TypeError("event must be SurfaceEventPassEvidence.")
    if not isinstance(surface, PreparedMultiRegionSurface):
        raise TypeError("surface must be PreparedMultiRegionSurface.")
    if not isinstance(state, MultiRegionSurfaceState):
        raise TypeError("state must be MultiRegionSurfaceState.")
    lineage, target_epoch = event.lineage, event.target_epoch
    if not event.committed or lineage is None or target_epoch is None:
        raise ValueError("Border re-preparation needs a committed event pass.")
    origin = multiregion_topology_epoch(source.surface.topology, source.positions_m)
    if event.source_epoch.epoch_id != origin.epoch_id:
        raise ValueError(
            "The event pass did not start from this border network's epoch and geometry."
        )
    state.require_topology(surface.topology)
    reached = multiregion_topology_epoch(surface.topology, state.positions)
    if target_epoch.epoch_id != reached.epoch_id:
        raise ValueError(
            "The target surface is not the committed geometry of this event pass."
        )
    return lineage, target_epoch


def _require_border_transport_rule(event: SurfaceEventPassEvidence, /) -> None:
    """Refuse accepted events whose border content has no declared transport."""
    for record in event.records:
        if not record.accepted:
            continue
        match record.kind:
            case (
                SurfaceEventKind.SPLIT | SurfaceEventKind.COLLAPSE | SurfaceEventKind.FLIP
            ):
                # Local remeshing: border identity is checked edge by edge.
                continue
            case (
                SurfaceEventKind.T1_POP
                | SurfaceEventKind.PINCH
                | SurfaceEventKind.MERGE
                | SurfaceEventKind.REGION_SPLIT
                | SurfaceEventKind.BURST
            ):
                raise PlateauBorderTransportError(
                    f"{record.kind.name} events have no declared physical transport "
                    "of Plateau-border liquid and surfactant; content is not "
                    "redistributed from lineage."
                )
            case unknown:
                assert_never(unknown)


def _border_keys(prepared: PreparedPlateauBorder, /) -> dict[tuple[int, int], int]:
    """Border axis index of every physical border keyed by stable vertex IDs."""
    ids = np.asarray(prepared.surface.topology.vertex_global_ids, dtype=np.int64)
    edges = np.asarray(prepared.border_edges[: prepared.border_count], dtype=np.int64)
    keys = np.sort(ids[edges], axis=1).tolist()
    return {
        (int(first), int(second)): index for index, (first, second) in enumerate(keys)
    }


def _border_parent(
    key: tuple[int, int],
    source_borders: dict[tuple[int, int], int],
    created: dict[int, tuple[int, ...]],
    /,
) -> int:
    """Source border holding the content of target border ``key``."""
    retained = source_borders.get(key)
    if retained is not None:
        return retained
    new = [vertex for vertex in key if vertex in created]
    if len(new) == 1:
        split = created[new[0]]
        other = key[1] if key[0] == new[0] else key[0]
        parent = source_borders.get((split[0], split[-1])) if len(split) == 2 else None
        if parent is not None and other in split:
            return parent
    raise PlateauBorderTransportError(
        f"Target Plateau border {key} has no physical content source; only "
        "retained borders and the two children of a split border are transported."
    )


def _border_event_routes(
    source: PreparedPlateauBorder,
    target: PreparedPlateauBorder,
    lineage: MultiRegionSurfaceLineage,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retained and split-border routes with child-length weights."""
    source_borders = _border_keys(source)
    target_borders = _border_keys(target)
    topology = source.surface.topology
    existing = set(
        np.asarray(topology.vertex_global_ids[: topology.vertex_count]).tolist()
    )
    created = {
        child: tuple(sorted(parents))
        for child, parents in lineage.vertex_parents
        if child not in existing
    }
    parents = np.asarray(
        [_border_parent(key, source_borders, created) for key in target_borders],
        dtype=np.int64,
    )
    children = np.asarray(list(target_borders.values()), dtype=np.int64)
    fan_out = np.bincount(parents, minlength=source.border_count)
    expected = np.asarray(
        [1 if key in target_borders else 2 for key in source_borders], dtype=np.int64
    )
    if np.any(fan_out != expected):
        lost = [
            key
            for key, index in source_borders.items()
            if fan_out[index] != expected[index]
        ]
        raise PlateauBorderTransportError(
            f"Source Plateau borders {lost} have no complete physical image; "
            "border coarsening and partial splits have no declared transport."
        )
    length = np.asarray(target._geometry()[0], dtype=np.float64)[children]
    total = np.bincount(parents, weights=length, minlength=source.border_count)
    return parents, children, length / total[parents]


def _preparation_evidence(
    plan: PlateauBorderPlan,
    topology: MultiRegionSurfaceTopology,
    status: PlateauBorderPreparationStatus,
    border_count: int,
    quad_count: int,
    incomplete: int,
    /,
) -> PlateauBorderPreparationEvidence:
    payload = {
        "kind": "plateau-border-preparation-evidence",
        "plan": plan.plan_id,
        "topology": topology.topology_id,
        "status": int(status),
        "border_count": border_count,
        "quad_count": quad_count,
        "incomplete": incomplete,
    }
    return PlateauBorderPreparationEvidence(
        status=status,
        accepted=status is PlateauBorderPreparationStatus.ACCEPTED,
        required_border_edges=border_count,
        border_edge_capacity=plan.border_edge_capacity,
        required_quad_points=quad_count,
        quad_point_capacity=plan.quad_point_capacity,
        incomplete_border_edge_count=incomplete,
        topology_id=topology.topology_id,
        resource_id=plan.resource_id,
        evidence_id=canonical_fingerprint(payload),
    )


def _resolve_status(*conditions: tuple[PlateauBorderStatus, Array]) -> Array:
    status = jnp.asarray(int(PlateauBorderStatus.ACCEPTED), dtype=jnp.int32)
    for value, condition in reversed(conditions):
        status = jnp.where(condition, int(value), status)
    return status


def _positive_scalar(value: ArrayLike, name: str, /) -> Array:
    host = np.asarray(value, dtype=np.float64)
    if host.shape != () or not np.isfinite(host) or host <= 0.0:
        raise ValueError(f"{name} must be one finite positive scalar.")
    return jnp.asarray(host)


__all__ = [
    "apply_foam_rupture_with_borders",
    "PlateauBorderAdaptation",
    "PlateauBorderBoundaryFlux",
    "PlateauBorderEvidence",
    "PlateauBorderPlan",
    "PlateauBorderPreparationError",
    "PlateauBorderPreparationEvidence",
    "PlateauBorderPreparationStatus",
    "PlateauBorderRates",
    "PlateauBorderResult",
    "PlateauBorderRimEvidence",
    "PlateauBorderRimResult",
    "PlateauBorderRimStatus",
    "PlateauBorderState",
    "PlateauBorderStatus",
    "PlateauBorderTransportError",
    "PreparedPlateauBorder",
]
