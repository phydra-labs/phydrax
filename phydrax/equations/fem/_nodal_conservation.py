#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization._cell_complex import (
    PolygonalConnectivity,
    PolyhedralConnectivity,
)
from ...discretization._conservation_boundary import (
    evaluate_conservation_boundary,
    PrescribedNormalFluxBoundary,
)
from ...discretization._conservation_policy import (
    DifferentiabilityPolicy,
    validate_differentiability_policy,
)
from ...discretization.fem._boundary import FiniteElementBoundarySet
from ...discretization.fem._generic import (
    FiniteElementDiscretization,
    IntegrationDomain,
)
from ...discretization.fem._geometry_quality import (
    finite_element_geometry_quality,
    FiniteElementGeometryQualityEvidence,
)
from ...discretization.fem._mortar import (
    serial_finite_element_mortar_plan,
)
from ...discretization.fem._reference_operator import _map_edge_rule, _map_face_rule
from ...discretization.fem._reference_topology import reference_cell_topology
from ...discretization.finite_volume._riemann import (
    AbstractArbitraryNormalNumericalFluxPlan,
)
from .._entropy_pair import ConvexEntropyPair
from .._finite_element_variational import (
    CellResidualAction,
    CompiledFiniteElementProblem,
    ExteriorFacetAction,
    FiniteElementExecutionContext,
    FiniteElementExecutionPolicy,
    FiniteElementForm,
    InteriorFacetAction,
)
from .._hyperbolic_systems import AbstractEntropyDiffusionSystem
from ._entropy_stability import (
    boundary_entropy_evidence,
    entropy_mortar_evidence,
    EntropyReferenceOperator,
    EntropyStableDGPlan,
    prepare_entropy_reference_operator,
)
from ._mass_inverse import PreparedDiscontinuousMassInverse
from ._operators import FiniteElementFacetMetricData, FiniteElementMetricData
from ._quadrature import QuadratureAccuracyPolicy, QuadratureEvidence
from ._trace_routes import (
    batch_dg_boundary_routes,
    batch_dg_mortar_routes,
    PreparedDGBoundaryBatch,
    PreparedDGMortarBatch,
    PreparedDGTraceRoute,
)
from ._viscous_conservation import ViscousDGPlan
from ._well_balanced import WellBalancedEquilibriumPlan


class NodalDGConservationMethodPlan(StrictModule, NonTrainableState):
    interface_flux: AbstractArbitraryNormalNumericalFluxPlan
    volume_quadrature: QuadratureAccuracyPolicy
    interior_facet_quadrature: QuadratureAccuracyPolicy
    exterior_facet_quadrature: QuadratureAccuracyPolicy
    viscous: ViscousDGPlan | None
    entropy_stability: EntropyStableDGPlan | None
    equilibrium: WellBalancedEquilibriumPlan | None
    accumulation: str = eqx.field(static=True)
    differentiability: DifferentiabilityPolicy = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        interface_flux: AbstractArbitraryNormalNumericalFluxPlan,
        /,
        *,
        volume_quadrature: QuadratureAccuracyPolicy | None = None,
        interior_facet_quadrature: QuadratureAccuracyPolicy | None = None,
        exterior_facet_quadrature: QuadratureAccuracyPolicy | None = None,
        viscous: ViscousDGPlan | None = None,
        entropy_stability: EntropyStableDGPlan | None = None,
        equilibrium: WellBalancedEquilibriumPlan | None = None,
        accumulation: str = "deterministic",
        differentiability: DifferentiabilityPolicy = "branchwise",
    ):
        if not isinstance(interface_flux, AbstractArbitraryNormalNumericalFluxPlan):
            raise TypeError("Nodal DG requires an arbitrary-normal interface flux.")
        volume = (
            QuadratureAccuracyPolicy("overintegrated")
            if volume_quadrature is None
            else volume_quadrature
        )
        interior = (
            volume if interior_facet_quadrature is None else interior_facet_quadrature
        )
        exterior = (
            volume if exterior_facet_quadrature is None else exterior_facet_quadrature
        )
        if not all(
            isinstance(value, QuadratureAccuracyPolicy)
            for value in (volume, interior, exterior)
        ):
            raise TypeError("Nodal DG quadrature values must be accuracy policies.")
        if viscous is not None and not isinstance(viscous, ViscousDGPlan):
            raise TypeError("viscous must be ViscousDGPlan or None.")
        if entropy_stability is not None and not isinstance(
            entropy_stability, EntropyStableDGPlan
        ):
            raise TypeError("entropy_stability must be EntropyStableDGPlan or None.")
        if equilibrium is not None and not isinstance(
            equilibrium, WellBalancedEquilibriumPlan
        ):
            raise TypeError("equilibrium must be WellBalancedEquilibriumPlan or None.")
        accumulation_ = str(accumulation)
        if accumulation_ not in ("fast", "deterministic", "compensated"):
            raise ValueError("Unknown nodal DG accumulation policy.")
        differentiability_ = validate_differentiability_policy(differentiability)
        self.interface_flux = interface_flux
        self.volume_quadrature = volume
        self.interior_facet_quadrature = interior
        self.exterior_facet_quadrature = exterior
        self.viscous = viscous
        self.entropy_stability = entropy_stability
        self.equilibrium = equilibrium
        self.accumulation = accumulation_
        self.differentiability = differentiability_
        self.method_id = canonical_fingerprint(
            {
                "kind": "nodal-dg-conservation-method",
                "interface_flux": interface_flux.flux_id,
                "volume_quadrature": volume.policy_id,
                "interior_quadrature": interior.policy_id,
                "exterior_quadrature": exterior.policy_id,
                "viscous": None if viscous is None else viscous.plan_id,
                "entropy_stability": (
                    None if entropy_stability is None else entropy_stability.plan_id
                ),
                "equilibrium": None if equilibrium is None else equilibrium.plan_id,
                "mass": "exact-cell-local",
                "accumulation": accumulation_,
                "entropy_evidence": "uncertified",
                "differentiability": differentiability_,
            }
        )


class NodalDGPreparationReport(StrictModule, NonTrainableState):
    volume_quadrature: QuadratureEvidence
    interior_quadrature: QuadratureEvidence
    exterior_quadrature: QuadratureEvidence
    mass_evidence_id: str = eqx.field(static=True)
    compilation_id: str = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    def __init__(
        self,
        volume_quadrature: QuadratureEvidence,
        interior_quadrature: QuadratureEvidence,
        exterior_quadrature: QuadratureEvidence,
        mass_evidence_id: str,
        compilation_id: str,
        /,
    ):
        self.volume_quadrature = volume_quadrature
        self.interior_quadrature = interior_quadrature
        self.exterior_quadrature = exterior_quadrature
        self.mass_evidence_id = str(mass_evidence_id)
        self.compilation_id = str(compilation_id)
        self.report_id = canonical_fingerprint(
            {
                "kind": "nodal-dg-preparation-report",
                "quadrature": (
                    volume_quadrature.evidence_id,
                    interior_quadrature.evidence_id,
                    exterior_quadrature.evidence_id,
                ),
                "mass": self.mass_evidence_id,
                "compilation": self.compilation_id,
            }
        )


class NodalDGConservationDiagnostics(StrictModule, NonTrainableState):
    total_integral: Array
    conservation_rate: Array
    boundary_flux_rate: Array
    source_integral: Array
    conservation_balance_defect: Array
    interface_entropy_production: Array | None
    boundary_entropy_defect: Array | None
    admissible: Array | None
    method_id: str = eqx.field(static=True)


class NodalDGFaceFluxes(StrictModule, NonTrainableState):
    normal_fluxes: tuple[Array, ...]
    signal_speeds: tuple[Array, ...]
    physical_weights: tuple[Array, ...]
    integrated_fluxes: tuple[Array, ...]
    entropy_productions: tuple[Array, ...]
    route_kinds: tuple[str, ...] = eqx.field(static=True)
    route_ids: tuple[str, ...] = eqx.field(static=True)


class NodalDGStableStepEvidence(StrictModule, NonTrainableState):
    step: Array
    maximum_advective_rate: Array
    maximum_diffusive_rate: Array
    minimum_cell_length: Array
    polynomial_degree: int = eqx.field(static=True)
    cfl: float = eqx.field(static=True)
    positive: Array
    method_id: str = eqx.field(static=True)


def _selected_degree(
    policy: QuadratureAccuracyPolicy,
    order: int,
    coordinate_order: int,
    /,
) -> int:
    return policy.resolve_degree(
        order,
        order,
        coordinate_order=coordinate_order,
        coefficient_order=0,
        kernel_polynomial_degree=None,
    )


def _same_block_interior_domain(
    discretization: FiniteElementDiscretization, /
) -> IntegrationDomain | None:
    if discretization.mesh.topological_dimension == 2 or isinstance(
        discretization.mesh.connectivity, PolyhedralConnectivity
    ):
        return None
    domain = discretization.interior_facet_domain
    owners = np.asarray(domain.owner_cells, dtype=np.int32)
    neighbours = np.asarray(domain.neighbour_cells, dtype=np.int32)
    offsets = np.cumsum(
        (0,) + tuple(block.cell_count for block in discretization.mesh.blocks)
    )
    owner_blocks = np.searchsorted(offsets[1:], owners, side="right")
    neighbour_blocks = np.searchsorted(offsets[1:], neighbours, side="right")
    rows = np.flatnonzero(owner_blocks == neighbour_blocks)
    if rows.size == 0:
        return None
    entity_indices = np.asarray(domain.entity_indices)[rows]
    return IntegrationDomain(
        "interior_facet",
        entity_indices,
        domain.support_id,
        domain.entity_set_id,
        owner_cells=owners[rows],
        neighbour_cells=neighbours[rows],
        owner_local_entities=np.asarray(domain.owner_local_entities)[rows],
        neighbour_local_entities=np.asarray(domain.neighbour_local_entities)[rows],
        neighbour_trace_permutations=np.asarray(domain.neighbour_trace_permutations)[
            rows
        ],
        selection_id=canonical_fingerprint(
            {
                "kind": "nodal-dg-same-block-interior",
                "domain": domain.domain_id,
                "entities": tuple(int(value) for value in entity_indices),
            }
        ),
    )


def _trace_nodes(element, local_facet: int, /) -> tuple[np.ndarray, np.ndarray]:
    topology = reference_cell_topology(element.cell_kind)
    vertex_ids = topology.entities[1][int(local_facet)]
    start = np.asarray(topology.vertices[vertex_ids[0]], dtype=float)
    stop = np.asarray(topology.vertices[vertex_ids[1]], dtype=float)
    tangent = stop - start
    length_squared = float(tangent @ tangent)
    nodes = np.asarray(element.reference_nodes, dtype=float)
    parameter = ((nodes - start) @ tangent) / length_squared
    projection = start + parameter[:, None] * tangent
    distance = np.max(np.abs(nodes - projection), axis=1)
    selected = np.flatnonzero(
        (distance <= 2.0e-10) & (parameter >= -2.0e-10) & (parameter <= 1.0 + 2.0e-10)
    )
    order = np.argsort(parameter[selected])
    return selected[order].astype(np.int32), parameter[selected][order]


def _edge_coordinate_trace(
    discretization: FiniteElementDiscretization,
    block_index: int,
    cell_index: int,
    local_facet: int,
    parameters: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    topology = reference_cell_topology(discretization.mesh.blocks[block_index].cell_kind)
    start_vertex, stop_vertex = topology.entities[1][int(local_facet)]
    start = np.asarray(topology.vertices[start_vertex], dtype=float)
    stop = np.asarray(topology.vertices[stop_vertex], dtype=float)
    values = np.asarray(parameters, dtype=float).reshape((-1,))
    reference_points = (1.0 - values)[:, None] * start[None, :] + values[:, None] * stop[
        None, :
    ]
    coordinate_element = discretization.coordinate_elements[block_index]
    basis, gradients = coordinate_element.tabulate(jnp.asarray(reference_points))
    coordinate_routes = discretization.coordinate_dofs[block_index][cell_index]
    coordinates = np.asarray(
        discretization.default_runtime.coordinates[coordinate_routes]
    )
    physical_points = np.asarray(basis) @ coordinates
    reference_tangent = stop - start
    physical_tangent = ein.contract(
        "qid,d,ia->qa",
        np.asarray(gradients),
        reference_tangent,
        coordinates,
    )
    return physical_points, np.asarray(physical_tangent)


def _mixed_mortar_routes(
    discretization: FiniteElementDiscretization,
    field_index: int,
    facet_degrees: dict[str, int],
    /,
) -> tuple[PreparedDGTraceRoute, ...]:
    from ...integration._rules import (
        GaussLegendreRule,
        GaussLobattoLegendreRule,
        ReferenceIntervalRule,
    )

    if discretization.mesh.topological_dimension != 2:
        return ()
    connectivity = discretization.mesh.connectivity
    if not isinstance(connectivity, PolygonalConnectivity):
        raise TypeError("Two-dimensional nodal DG requires polygonal connectivity.")
    offsets = np.cumsum(
        (0,) + tuple(block.cell_count for block in discretization.mesh.blocks)
    )
    domain = discretization.interior_facet_domain
    routes = []
    mortar_cache = {}
    dof_map = discretization.dof_maps[field_index]
    for row, (facet, owner, neighbour, owner_local, neighbour_local) in enumerate(
        zip(
            np.asarray(domain.entity_indices, dtype=np.int32),
            np.asarray(domain.owner_cells, dtype=np.int32),
            np.asarray(domain.neighbour_cells, dtype=np.int32),
            np.asarray(domain.owner_local_entities, dtype=np.int32),
            np.asarray(domain.neighbour_local_entities, dtype=np.int32),
            strict=True,
        )
    ):
        owner_block = int(np.searchsorted(offsets[1:], owner, side="right"))
        neighbour_block = int(np.searchsorted(offsets[1:], neighbour, side="right"))
        owner_cell = int(owner - offsets[owner_block])
        neighbour_cell = int(neighbour - offsets[neighbour_block])
        owner_element = discretization.elements[field_index][owner_block]
        neighbour_element = discretization.elements[field_index][neighbour_block]
        owner_trace, owner_nodes = _trace_nodes(owner_element, int(owner_local))
        neighbour_trace, neighbour_nodes = _trace_nodes(
            neighbour_element, int(neighbour_local)
        )
        degree = max(
            facet_degrees[discretization.mesh.blocks[owner_block].name],
            facet_degrees[discretization.mesh.blocks[neighbour_block].name],
        )
        quadrature = ReferenceIntervalRule(
            GaussLegendreRule(_rule_count(degree))
        ).materialize()
        quadrature_points = np.asarray(quadrature.points)
        quadrature_weights = np.asarray(quadrature.weights)
        mortar_nodes = np.asarray(
            ReferenceIntervalRule(
                GaussLobattoLegendreRule(
                    max(owner_element.degree, neighbour_element.degree) + 1
                )
            )
            .materialize()
            .points
        )
        owner_sign = float(np.asarray(connectivity.cell_edge_signs)[owner, owner_local])
        neighbour_sign = float(
            np.asarray(connectivity.cell_edge_signs)[neighbour, neighbour_local]
        )
        reversed_orientation = owner_sign * neighbour_sign < 0.0
        right_points = (
            1.0 - quadrature_points if reversed_orientation else quadrature_points
        )
        owner_physical, owner_tangent = _edge_coordinate_trace(
            discretization,
            owner_block,
            owner_cell,
            int(owner_local),
            quadrature_points,
        )
        neighbour_physical, neighbour_tangent = _edge_coordinate_trace(
            discretization,
            neighbour_block,
            neighbour_cell,
            int(neighbour_local),
            right_points,
        )
        if reversed_orientation:
            neighbour_tangent = -neighbour_tangent
        coordinate_scale = max(
            1.0,
            float(np.max(np.abs(owner_physical))),
            float(np.max(np.abs(neighbour_physical))),
        )
        coordinate_tolerance = 1.0e-9 * coordinate_scale
        coordinate_defect = float(np.max(np.abs(owner_physical - neighbour_physical)))
        tangent_defect = float(np.max(np.abs(owner_tangent - neighbour_tangent)))
        if max(coordinate_defect, tangent_defect) > coordinate_tolerance:
            raise ValueError(
                "Mixed mortar coordinate traces are not watertight and tangent "
                "compatible."
            )
        physical_points = 0.5 * (owner_physical + neighbour_physical)
        tangent = 0.5 * (owner_tangent + neighbour_tangent)
        measure = np.sqrt(np.sum(tangent * tangent, axis=-1))
        if np.any(~np.isfinite(measure)) or np.any(measure <= 0.0):
            raise ValueError("Mixed mortar coordinate measure must be positive.")
        normal = np.stack((tangent[:, 1], -tangent[:, 0]), axis=-1)
        normal = normal / measure[:, None]
        owner_coordinate_routes = discretization.coordinate_dofs[owner_block][owner_cell]
        owner_center = np.mean(
            np.asarray(discretization.default_runtime.coordinates)[
                owner_coordinate_routes
            ],
            axis=0,
        )
        if (
            float(
                np.mean(normal, axis=0)
                @ (np.mean(physical_points, axis=0) - owner_center)
            )
            < 0.0
        ):
            normal = -normal
        mortar_key = (
            owner_element.element_id,
            neighbour_element.element_id,
            reversed_orientation,
            tuple(float(value) for value in np.round(quadrature_weights * measure, 14)),
        )
        if mortar_key in mortar_cache:
            mortar = mortar_cache[mortar_key]
        else:
            mortar = serial_finite_element_mortar_plan(
                owner_nodes[:, None],
                neighbour_nodes[:, None],
                mortar_nodes,
                quadrature_points,
                quadrature_weights,
                left_evaluation_points=quadrature_points,
                right_evaluation_points=right_points,
                left_physical_coordinates=owner_physical,
                right_physical_coordinates=neighbour_physical,
                coordinate_measure=measure,
                coordinate_tolerance=coordinate_tolerance,
                declared_reproduction_degree=min(
                    owner_element.degree, neighbour_element.degree
                ),
                left_polynomial_coordinates=owner_nodes[:, None],
                right_polynomial_coordinates=(
                    1.0 - neighbour_nodes[:, None]
                    if reversed_orientation
                    else neighbour_nodes[:, None]
                ),
                mortar_polynomial_coordinates=mortar_nodes,
                polynomial_evaluation_points=quadrature_points,
                interface_id=canonical_fingerprint(
                    {
                        "kind": "mixed-nodal-dg-mortar-signature",
                        "owner_element": owner_element.element_id,
                        "neighbour_element": neighbour_element.element_id,
                        "reversed": reversed_orientation,
                        "weights": mortar_key[-1],
                    }
                ),
            )
            mortar_cache[mortar_key] = mortar
        owner_dofs = np.asarray(dof_map.cell_dofs[owner_block][owner_cell])[owner_trace]
        neighbour_dofs = np.asarray(dof_map.cell_dofs[neighbour_block][neighbour_cell])[
            neighbour_trace
        ]
        route_id = canonical_fingerprint(
            {
                "kind": "nodal-dg-mortar-route",
                "row": row,
                "facet": int(facet),
                "mortar": mortar.plan_id,
            }
        )
        routes.append(
            PreparedDGTraceRoute(
                "mortar",
                jnp.asarray(owner_dofs, dtype=jnp.int32),
                neighbour_dofs=jnp.asarray(neighbour_dofs, dtype=jnp.int32),
                normal=jnp.asarray(normal),
                mortar=mortar,
                route_id=route_id,
            )
        )
    return tuple(routes)


def _periodic_trace_routes(
    discretization: FiniteElementDiscretization,
    field_index: int,
    boundaries: FiniteElementBoundarySet,
    facet_degrees: dict[str, int],
    component_count: int,
    /,
) -> tuple[PreparedDGTraceRoute, ...]:
    from ...integration._rules import GaussLegendreRule, ReferenceIntervalRule

    if not boundaries.periodic_pairs:
        return ()
    if discretization.mesh.topological_dimension != 2:
        raise ValueError(
            "Transformed periodic trace preparation currently requires edge facets."
        )
    exterior = discretization.exterior_facet_domain
    facet_positions = {
        int(facet): index
        for index, facet in enumerate(np.asarray(exterior.entity_indices, dtype=np.int32))
    }
    offsets = np.cumsum(
        (0,) + tuple(block.cell_count for block in discretization.mesh.blocks)
    )
    dof_map = discretization.dof_maps[field_index]
    routes = []
    for pair in boundaries.periodic_pairs:
        if pair.transform is None:
            raise ValueError(
                "Nodal DG periodic pairs require an explicit periodic transform."
            )
        if pair.transform.orientation.shape != "edge":
            raise ValueError("Two-dimensional periodic facets require edge orientation.")
        owner_position = facet_positions[pair.owner_facet]
        neighbour_position = facet_positions[pair.neighbour_facet]
        owner = int(np.asarray(exterior.owner_cells)[owner_position])
        neighbour = int(np.asarray(exterior.owner_cells)[neighbour_position])
        owner_local = int(np.asarray(exterior.owner_local_entities)[owner_position])
        neighbour_local = int(
            np.asarray(exterior.owner_local_entities)[neighbour_position]
        )
        owner_block = int(np.searchsorted(offsets[1:], owner, side="right"))
        neighbour_block = int(np.searchsorted(offsets[1:], neighbour, side="right"))
        owner_cell = int(owner - offsets[owner_block])
        neighbour_cell = int(neighbour - offsets[neighbour_block])
        owner_element = discretization.elements[field_index][owner_block]
        neighbour_element = discretization.elements[field_index][neighbour_block]
        degree = max(
            facet_degrees[discretization.mesh.blocks[owner_block].name],
            facet_degrees[discretization.mesh.blocks[neighbour_block].name],
        )
        quadrature = ReferenceIntervalRule(
            GaussLegendreRule(_rule_count(degree))
        ).materialize()
        owner_parameter = np.asarray(quadrature.points)
        neighbour_parameter = (
            1.0 - owner_parameter
            if pair.transform.orientation.permutation == (1, 0)
            else owner_parameter
        )
        owner_physical, owner_tangent = _edge_coordinate_trace(
            discretization,
            owner_block,
            owner_cell,
            owner_local,
            owner_parameter,
        )
        neighbour_physical, neighbour_tangent = _edge_coordinate_trace(
            discretization,
            neighbour_block,
            neighbour_cell,
            neighbour_local,
            neighbour_parameter,
        )
        if pair.transform.orientation.permutation == (1, 0):
            neighbour_tangent = -neighbour_tangent
        mapped_owner = np.asarray(
            pair.transform.map_coordinates(jnp.asarray(owner_physical))
        )
        mapped_tangent = ein.contract(
            "ij,qj->qi",
            np.asarray(pair.transform.coordinate_matrix),
            owner_tangent,
        )
        defect = max(
            float(np.max(np.abs(mapped_owner - neighbour_physical))),
            float(np.max(np.abs(mapped_tangent - neighbour_tangent))),
        )
        scale = max(1.0, float(np.max(np.abs(neighbour_physical))))
        if defect > pair.transform.tolerance * scale:
            raise ValueError("Periodic coordinate traces are incompatible.")
        owner_topology = reference_cell_topology(
            discretization.mesh.blocks[owner_block].cell_kind
        )
        neighbour_topology = reference_cell_topology(
            discretization.mesh.blocks[neighbour_block].cell_kind
        )
        owner_vertices = owner_topology.entities[1][owner_local]
        neighbour_vertices = neighbour_topology.entities[1][neighbour_local]
        owner_start = jnp.asarray(owner_topology.vertices[owner_vertices[0]])
        owner_stop = jnp.asarray(owner_topology.vertices[owner_vertices[1]])
        neighbour_start = jnp.asarray(neighbour_topology.vertices[neighbour_vertices[0]])
        neighbour_stop = jnp.asarray(neighbour_topology.vertices[neighbour_vertices[1]])
        owner_points = (1.0 - jnp.asarray(owner_parameter)) * owner_start + jnp.asarray(
            owner_parameter
        ) * owner_stop
        neighbour_points = (
            1.0 - jnp.asarray(neighbour_parameter)
        ) * neighbour_start + jnp.asarray(neighbour_parameter) * neighbour_stop
        owner_basis = owner_element.tabulate(owner_points)[0]
        neighbour_basis = neighbour_element.tabulate(neighbour_points)[0]
        measure = np.sqrt(np.sum(owner_tangent * owner_tangent, axis=-1))
        normal = np.stack((owner_tangent[:, 1], -owner_tangent[:, 0]), axis=-1)
        normal = normal / measure[:, None]
        owner_coordinates = discretization.default_runtime.coordinates[
            discretization.coordinate_dofs[owner_block][owner_cell]
        ]
        owner_center = np.mean(np.asarray(owner_coordinates), axis=0)
        if (
            float(
                np.mean(normal, axis=0) @ (np.mean(owner_physical, axis=0) - owner_center)
            )
            < 0.0
        ):
            normal = -normal
        component_transform = np.asarray(pair.transform.component_matrix)
        if component_transform.shape == (1, 1):
            component_transform = np.eye(component_count)
        if component_transform.shape != (component_count, component_count):
            raise ValueError("Periodic component transform has incompatible shape.")
        route_id = canonical_fingerprint(
            {
                "kind": "nodal-dg-periodic-route",
                "pair": pair.pair_id,
                "owner_element": owner_element.element_id,
                "neighbour_element": neighbour_element.element_id,
            }
        )
        routes.append(
            PreparedDGTraceRoute(
                "periodic",
                dof_map.cell_dofs[owner_block][owner_cell],
                neighbour_dofs=dof_map.cell_dofs[neighbour_block][neighbour_cell],
                owner_basis=owner_basis,
                neighbour_basis=neighbour_basis,
                physical_points=jnp.asarray(owner_physical),
                physical_weights=jnp.asarray(quadrature.weights * measure),
                normal=jnp.asarray(normal),
                component_transform=jnp.asarray(component_transform),
                coordinate_transform=pair.transform.coordinate_matrix,
                route_id=route_id,
            )
        )
    return tuple(routes)


def _hybrid_boundary_routes(
    discretization: FiniteElementDiscretization,
    field_index: int,
    boundaries: FiniteElementBoundarySet,
    facet_degree: int,
    /,
) -> tuple[PreparedDGTraceRoute, ...]:
    from ...integration._rules import (
        GaussLegendreRule,
        reference_rule_data,
        ReferenceIntervalRule,
        ReferenceQuadrilateralRule,
        ReferenceTriangleRule,
    )

    dimension = discretization.mesh.topological_dimension
    if dimension not in (1, 2, 3):
        return ()
    if dimension == 3 and any(
        block.cell_kind not in ("tetrahedron", "hexahedron", "prism", "pyramid")
        for block in discretization.mesh.blocks
    ):
        return ()
    policy_by_facet = {
        int(facet): patch.boundary
        for patch in boundaries.patches
        for facet in np.asarray(patch.domain.entity_indices, dtype=np.int32)
    }
    exterior = discretization.exterior_facet_domain
    dof_map = discretization.dof_maps[field_index]
    offsets = np.cumsum(
        (0,) + tuple(block.cell_count for block in discretization.mesh.blocks)
    )
    axis_rule = GaussLegendreRule(_rule_count(facet_degree))
    routes = []
    for facet, owner, local_facet in zip(
        np.asarray(exterior.entity_indices, dtype=np.int32),
        np.asarray(exterior.owner_cells, dtype=np.int32),
        np.asarray(exterior.owner_local_entities, dtype=np.int32),
        strict=True,
    ):
        block_index = int(np.searchsorted(offsets[1:], owner, side="right"))
        if int(facet) not in policy_by_facet:
            continue
        local_cell = int(owner - offsets[block_index])
        block = discretization.mesh.blocks[block_index]
        element = discretization.elements[field_index][block_index]
        coordinate_element = discretization.coordinate_elements[block_index]
        coordinate_routes = discretization.coordinate_dofs[block_index]
        topology = reference_cell_topology(block.cell_kind)
        if dimension == 1:
            points = jnp.asarray(
                (topology.vertices[int(local_facet)],), dtype=jnp.float64
            )
            basis_values, basis_gradients = element.tabulate(points)
            coordinate_basis, _coordinate_gradients = coordinate_element.tabulate(points)
            coordinates = discretization.default_runtime.coordinates[
                coordinate_routes[local_cell]
            ]
            physical_points = ein.contract(
                "qi,id->qd", coordinate_basis, coordinates, backend="jax"
            )
            physical_weights = jnp.ones((1,), dtype=physical_points.dtype)
            normal = jnp.asarray(
                ((-1.0,),) if int(local_facet) == 0 else ((1.0,),),
                dtype=physical_points.dtype,
            )
            physical_gradients = basis_gradients
        elif dimension == 2:
            data = reference_rule_data(ReferenceIntervalRule(axis_rule))
            points, weights, normals = _map_edge_rule(
                block.cell_kind, int(local_facet), data
            )
            coordinate_basis, coordinate_gradients = coordinate_element.tabulate(points)
            local_coordinates = discretization.default_runtime.coordinates[
                coordinate_routes[local_cell]
            ][None, ...]
            metric = FiniteElementMetricData(
                coordinate_basis,
                coordinate_gradients,
                local_coordinates,
                weights,
            )
            facet_metric = FiniteElementFacetMetricData(metric, normals, weights)
            basis_values, basis_gradients = element.tabulate(points)
            physical_gradients = metric.physical_gradients(basis_gradients)[0]
            physical_points = facet_metric.physical_points[0]
            physical_weights = facet_metric.physical_weights[0]
            normal = facet_metric.normal[0]
        else:
            face_arity = len(topology.entities[2][int(local_facet)])
            rule = (
                ReferenceTriangleRule(axis_rule)
                if face_arity == 3
                else ReferenceQuadrilateralRule(axis_rule)
            )
            data = reference_rule_data(rule)
            points, weights, normals = _map_face_rule(
                block.cell_kind, int(local_facet), data
            )
            coordinate_basis, coordinate_gradients = coordinate_element.tabulate(points)
            local_coordinates = discretization.default_runtime.coordinates[
                coordinate_routes[local_cell]
            ][None, ...]
            metric = FiniteElementMetricData(
                coordinate_basis,
                coordinate_gradients,
                local_coordinates,
                weights,
            )
            facet_metric = FiniteElementFacetMetricData(metric, normals, weights)
            basis_values, basis_gradients = element.tabulate(points)
            physical_gradients = metric.physical_gradients(basis_gradients)[0]
            physical_points = facet_metric.physical_points[0]
            physical_weights = facet_metric.physical_weights[0]
            normal = facet_metric.normal[0]
        route_id = canonical_fingerprint(
            {
                "kind": "nodal-dg-hybrid-boundary-route",
                "facet": int(facet),
                "owner": int(owner),
                "local_facet": int(local_facet),
                "boundary": policy_by_facet[int(facet)].boundary_id,
            }
        )
        routes.append(
            PreparedDGTraceRoute(
                "boundary",
                dof_map.cell_dofs[block_index][local_cell],
                owner_basis=basis_values,
                owner_gradients=physical_gradients,
                physical_points=physical_points,
                physical_weights=physical_weights,
                normal=normal,
                boundary=policy_by_facet[int(facet)],
                route_id=route_id,
            )
        )
    return tuple(routes)


def _three_dimensional_interface_routes(
    discretization: FiniteElementDiscretization,
    field_index: int,
    facet_degree: int,
    /,
) -> tuple[PreparedDGTraceRoute, ...]:
    from ...integration._rules import (
        GaussLegendreRule,
        reference_rule_data,
        ReferenceQuadrilateralRule,
        ReferenceTriangleRule,
    )

    connectivity = discretization.mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        return ()
    offsets = np.cumsum(
        (0,) + tuple(block.cell_count for block in discretization.mesh.blocks)
    )
    domain = discretization.interior_facet_domain
    axis_rule = GaussLegendreRule(_rule_count(facet_degree))
    routes = []
    dof_map = discretization.dof_maps[field_index]
    for facet, owner, neighbour, owner_local, neighbour_local in zip(
        np.asarray(domain.entity_indices, dtype=np.int32),
        np.asarray(domain.owner_cells, dtype=np.int32),
        np.asarray(domain.neighbour_cells, dtype=np.int32),
        np.asarray(domain.owner_local_entities, dtype=np.int32),
        np.asarray(domain.neighbour_local_entities, dtype=np.int32),
        strict=True,
    ):
        owner_block = int(np.searchsorted(offsets[1:], owner, side="right"))
        neighbour_block = int(np.searchsorted(offsets[1:], neighbour, side="right"))
        owner_cell = int(owner - offsets[owner_block])
        neighbour_cell = int(neighbour - offsets[neighbour_block])
        owner_block_data = discretization.mesh.blocks[owner_block]
        neighbour_block_data = discretization.mesh.blocks[neighbour_block]
        owner_topology = reference_cell_topology(owner_block_data.cell_kind)
        neighbour_topology = reference_cell_topology(neighbour_block_data.cell_kind)
        owner_face_vertices = owner_topology.entities[2][int(owner_local)]
        neighbour_face_vertices = neighbour_topology.entities[2][int(neighbour_local)]
        arity = len(owner_face_vertices)
        if len(neighbour_face_vertices) != arity:
            raise ValueError("Three-dimensional interface face arities disagree.")
        rule = (
            ReferenceTriangleRule(axis_rule)
            if arity == 3
            else ReferenceQuadrilateralRule(axis_rule)
        )
        data = reference_rule_data(rule)
        owner_points, weights, owner_normals = _map_face_rule(
            owner_block_data.cell_kind, int(owner_local), data
        )
        if arity == 3:
            first = np.asarray(data.points)[:, 0]
            second = np.asarray(data.points)[:, 1]
            barycentric = np.stack((1.0 - first - second, first, second), axis=-1)
        else:
            first = np.asarray(data.points)[:, 0]
            second = np.asarray(data.points)[:, 1]
            barycentric = np.stack(
                (
                    (1.0 - first) * (1.0 - second),
                    first * (1.0 - second),
                    first * second,
                    (1.0 - first) * second,
                ),
                axis=-1,
            )
        owner_global_vertices = np.asarray(owner_block_data.vertices)[owner_cell][
            np.asarray(owner_face_vertices)
        ]
        neighbour_global_vertices = np.asarray(neighbour_block_data.vertices)[
            neighbour_cell
        ][np.asarray(neighbour_face_vertices)]
        weights_by_vertex = {
            int(vertex): barycentric[:, index]
            for index, vertex in enumerate(owner_global_vertices)
        }
        neighbour_barycentric = np.stack(
            tuple(weights_by_vertex[int(vertex)] for vertex in neighbour_global_vertices),
            axis=-1,
        )
        neighbour_reference_vertices = np.asarray(
            tuple(neighbour_topology.vertices[index] for index in neighbour_face_vertices)
        )
        neighbour_points = jnp.asarray(
            neighbour_barycentric @ neighbour_reference_vertices
        )
        owner_coordinate_element = discretization.coordinate_elements[owner_block]
        owner_coordinate_routes = discretization.coordinate_dofs[owner_block][owner_cell]
        coordinate_basis, coordinate_gradients = owner_coordinate_element.tabulate(
            owner_points
        )
        metric = FiniteElementMetricData(
            coordinate_basis,
            coordinate_gradients,
            discretization.default_runtime.coordinates[owner_coordinate_routes][
                None, ...
            ],
            weights,
        )
        facet_metric = FiniteElementFacetMetricData(metric, owner_normals, weights)
        owner_element = discretization.elements[field_index][owner_block]
        neighbour_element = discretization.elements[field_index][neighbour_block]
        owner_basis = owner_element.tabulate(owner_points)[0]
        neighbour_basis = neighbour_element.tabulate(neighbour_points)[0]
        route_id = canonical_fingerprint(
            {
                "kind": "nodal-dg-three-dimensional-interface",
                "facet": int(facet),
                "owner_element": owner_element.element_id,
                "neighbour_element": neighbour_element.element_id,
            }
        )
        routes.append(
            PreparedDGTraceRoute(
                "conforming",
                dof_map.cell_dofs[owner_block][owner_cell],
                neighbour_dofs=dof_map.cell_dofs[neighbour_block][neighbour_cell],
                owner_basis=owner_basis,
                neighbour_basis=neighbour_basis,
                physical_points=facet_metric.physical_points[0],
                physical_weights=facet_metric.physical_weights[0],
                normal=facet_metric.normal[0],
                route_id=route_id,
            )
        )
    return tuple(routes)


def _rule_count(degree: int, /) -> int:
    return max(1, (int(degree) + 2) // 2)


def _rules(cell_kind: str, volume_degree: int, facet_degree: int, /):
    from ...integration._rules import (
        GaussLegendreRule,
        ReferenceHexahedronRule,
        ReferenceIntervalRule,
        ReferencePrismRule,
        ReferencePyramidRule,
        ReferenceQuadrilateralRule,
        ReferenceTetrahedronRule,
        ReferenceTriangleRule,
    )

    volume_axis = GaussLegendreRule(_rule_count(volume_degree))
    facet_axis = GaussLegendreRule(_rule_count(facet_degree))
    if cell_kind == "interval":
        return ReferenceIntervalRule(volume_axis), None
    if cell_kind == "triangle":
        return ReferenceTriangleRule(volume_axis), ReferenceIntervalRule(facet_axis)
    if cell_kind == "quadrilateral":
        return ReferenceQuadrilateralRule(volume_axis), ReferenceIntervalRule(facet_axis)
    if cell_kind == "tetrahedron":
        return ReferenceTetrahedronRule(volume_axis), ReferenceTriangleRule(facet_axis)
    if cell_kind == "hexahedron":
        return ReferenceHexahedronRule(volume_axis), ReferenceQuadrilateralRule(
            facet_axis
        )
    if cell_kind == "prism":
        return ReferencePrismRule(volume_axis), None
    if cell_kind == "pyramid":
        return ReferencePyramidRule(volume_axis), None
    raise ValueError("Unsupported Nodal DG cell kind.")


class PreparedNodalDGConservationDynamics(StrictModule):
    system: Any
    discretization: FiniteElementDiscretization
    method: NodalDGConservationMethodPlan
    boundaries: FiniteElementBoundarySet
    entropy_pair: ConvexEntropyPair | None
    source: Callable | None = eqx.field(static=True)
    compiled_finite_element_problem: CompiledFiniteElementProblem
    mass_inverse: PreparedDiscontinuousMassInverse
    mortar_routes: tuple[PreparedDGTraceRoute, ...]
    mortar_batches: tuple[PreparedDGMortarBatch, ...]
    periodic_routes: tuple[PreparedDGTraceRoute, ...]
    hybrid_boundary_routes: tuple[PreparedDGTraceRoute, ...]
    boundary_batches: tuple[PreparedDGBoundaryBatch, ...]
    three_dimensional_interface_routes: tuple[PreparedDGTraceRoute, ...]
    geometry_quality: FiniteElementGeometryQualityEvidence
    entropy_operators: tuple[EntropyReferenceOperator | None, ...]
    report: NodalDGPreparationReport
    dynamics_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: Any,
        discretization: FiniteElementDiscretization,
        method: NodalDGConservationMethodPlan,
        boundaries: FiniteElementBoundarySet,
        /,
        *,
        source: Callable | None = None,
        entropy_pair: ConvexEntropyPair | None = None,
    ):
        if not isinstance(discretization, FiniteElementDiscretization):
            raise TypeError("Nodal DG requires FiniteElementDiscretization.")
        if not isinstance(method, NodalDGConservationMethodPlan):
            raise TypeError("method must be NodalDGConservationMethodPlan.")
        if not isinstance(boundaries, FiniteElementBoundarySet):
            raise TypeError("Nodal DG requires exhaustive FiniteElementBoundarySet.")
        if source is not None and not callable(source):
            raise TypeError("Nodal DG source must be callable or None.")
        if entropy_pair is not None and not isinstance(entropy_pair, ConvexEntropyPair):
            raise TypeError("entropy_pair must be ConvexEntropyPair or None.")
        if len(discretization.field_spaces) != 1:
            raise ValueError("Nodal DG requires one conserved field.")
        blocks = discretization.mesh.blocks
        cell_kinds = tuple(block.cell_kind for block in blocks)
        if any(
            kind
            not in (
                "interval",
                "triangle",
                "quadrilateral",
                "tetrahedron",
                "hexahedron",
                "prism",
                "pyramid",
            )
            for kind in cell_kinds
        ):
            raise ValueError("Nodal DG encountered an unsupported cell kind.")
        if method.entropy_stability is not None:
            if entropy_pair is None:
                entropy_pair = method.entropy_stability.entropy_pair
            elif (
                entropy_pair.entropy_id
                != method.entropy_stability.entropy_pair.entropy_id
            ):
                raise ValueError(
                    "Nodal DG entropy pair and entropy-stability plan must match."
                )
            for patch in boundaries.patches:
                method.entropy_stability.boundary_contract(patch.boundary.boundary_id)
        if len(blocks) > 1:
            dimension = discretization.mesh.topological_dimension
            compatible = (
                dimension == 2
                and all(kind in ("triangle", "quadrilateral") for kind in cell_kinds)
            ) or (
                dimension == 3
                and all(
                    kind in ("tetrahedron", "hexahedron", "prism", "pyramid")
                    for kind in cell_kinds
                )
            )
            if not compatible:
                raise ValueError("Mixed nodal DG cell kinds are incompatible.")
        geometry_quality = finite_element_geometry_quality(discretization)
        if not bool(np.all(np.asarray(geometry_quality.valid_cells))):
            raise ValueError("Nodal DG geometry quality evidence failed.")
        if discretization.mesh.topological_dimension != system.dimension:
            raise ValueError("Nodal DG mesh and system dimensions must match.")
        field_name = discretization.field_spaces[0].name
        field_index = discretization._field_index(field_name)
        elements = discretization.elements[field_index]
        dof_map = discretization.dof_maps[field_index]
        if (
            any(
                element.conformity != "L2" or element.representation != "point_value"
                for element in elements
            )
            or dof_map.association != "cell"
        ):
            raise ValueError("Nodal DG requires cell-local point-value L2 elements.")
        if dof_map.component_shape != (system.component_count,):
            raise ValueError("Nodal DG component shape must match the system.")
        if method.viscous is not None:
            if not isinstance(system, AbstractEntropyDiffusionSystem):
                raise TypeError(
                    "Nodal DG viscosity requires AbstractEntropyDiffusionSystem."
                )
            for patch in boundaries.patches:
                method.viscous.boundary_closure(patch.boundary.boundary_id)
        elif isinstance(system, AbstractEntropyDiffusionSystem):
            raise ValueError("Entropy-diffusion nodal DG requires a viscous plan.")
        volume_degrees = {}
        interior_degrees = {}
        exterior_degrees = {}
        volume_rules = {}
        interior_rules = {}
        exterior_rules = {}
        for block_index, (block, element) in enumerate(
            zip(blocks, elements, strict=True)
        ):
            coordinate_order = discretization.coordinate_elements[block_index].degree
            volume_degrees[block.name] = _selected_degree(
                method.volume_quadrature, element.degree, coordinate_order
            )
            interior_degrees[block.name] = _selected_degree(
                method.interior_facet_quadrature, element.degree, coordinate_order
            )
            exterior_degrees[block.name] = _selected_degree(
                method.exterior_facet_quadrature, element.degree, coordinate_order
            )
            volume_rule, interior_rule = _rules(
                block.cell_kind,
                volume_degrees[block.name],
                interior_degrees[block.name],
            )
            _unused, exterior_rule = _rules(
                block.cell_kind,
                volume_degrees[block.name],
                exterior_degrees[block.name],
            )
            volume_rules[block.name] = volume_rule
            interior_rules[block.name] = interior_rule
            exterior_rules[block.name] = exterior_rule
        entropy_operators = []
        if method.entropy_stability is not None:
            from ...integration._rules import (
                GaussLegendreRule,
                ReferenceQuadrilateralRule,
                ReferenceTriangleRule,
            )

            for block, element in zip(blocks, elements, strict=True):
                topology = reference_cell_topology(block.cell_kind)
                if element.topological_dimension == 1:
                    entropy_operators.append(None)
                    continue
                if element.topological_dimension == 2:
                    facet_rules = (interior_rules[block.name],) * len(
                        topology.entities[1]
                    )
                elif interior_rules[block.name] is not None:
                    facet_rules = (interior_rules[block.name],) * len(
                        topology.entities[2]
                    )
                else:
                    axis_rule = GaussLegendreRule(
                        _rule_count(interior_degrees[block.name])
                    )
                    facet_rules = tuple(
                        ReferenceTriangleRule(axis_rule)
                        if len(face) == 3
                        else ReferenceQuadrilateralRule(axis_rule)
                        for face in topology.entities[2]
                    )
                entropy_operators.append(
                    prepare_entropy_reference_operator(
                        element,
                        volume_rules[block.name],
                        facet_rules,
                        tolerance=method.entropy_stability.tolerance,
                    )
                )
        entropy_operators = tuple(entropy_operators)
        same_block_domain = _same_block_interior_domain(discretization)
        mortar_routes = _mixed_mortar_routes(
            discretization, field_index, interior_degrees
        )
        mortar_batches = batch_dg_mortar_routes(mortar_routes)
        periodic_routes = _periodic_trace_routes(
            discretization,
            field_index,
            boundaries,
            interior_degrees,
            system.component_count,
        )
        hybrid_boundary_routes = _hybrid_boundary_routes(
            discretization,
            field_index,
            boundaries,
            max(exterior_degrees.values()),
        )

        boundary_batches = batch_dg_boundary_routes(hybrid_boundary_routes)
        three_dimensional_interface_routes = _three_dimensional_interface_routes(
            discretization,
            field_index,
            max(interior_degrees.values()),
        )

        def volume_kernel(
            values,
            gradients,
            points,
            physical_weights,
            test_basis,
            test_gradients,
            context,
        ):
            del gradients, points
            state = values[0]
            if method.entropy_stability is not None:
                return jnp.zeros(
                    (state.shape[0], test_basis.shape[-1], state.shape[-1]),
                    dtype=state.dtype,
                )
            del test_basis
            flux = jnp.stack(
                tuple(
                    system.physical_flux(state, axis, context.user_args)
                    for axis in range(system.dimension)
                ),
                axis=-1,
            )
            return -ein.contract(
                "cq,cqid,cqvd->civ",
                physical_weights,
                test_gradients,
                flux,
                backend="jax",
            )

        def interface_kernel(plus_values, minus_values, points, weights, normal, context):
            del points, weights
            plus = plus_values[0]
            minus = minus_values[0]
            numerical = method.interface_flux.normal_face_flux(
                system,
                plus,
                minus,
                normal,
                context.user_args,
            ).normal_flux
            if method.entropy_stability is None:
                return numerical, -numerical
            return (
                numerical - system.physical_normal_flux(plus, normal, context.user_args),
                -numerical
                + system.physical_normal_flux(minus, normal, context.user_args),
            )

        actions = [
            CellResidualAction(
                field_name,
                (field_name,),
                volume_kernel,
                domain=discretization.cell_domain,
                rules=tuple(volume_rules.items()),
                action_id="nodal-dg-volume",
            )
        ]
        if same_block_domain is not None:
            actions.append(
                InteriorFacetAction(
                    field_name,
                    (field_name,),
                    interface_kernel,
                    domain=same_block_domain,
                    rules=tuple(interior_rules.items()),
                    action_id="nodal-dg-same-block-interior-flux",
                )
            )
        custom_boundaries = bool(hybrid_boundary_routes)
        for patch in () if custom_boundaries else boundaries.patches:

            def boundary_kernel(boundary):
                def kernel(plus_values, points, weights, normal, context):
                    del weights
                    plus = plus_values[0]
                    trace = evaluate_conservation_boundary(
                        boundary,
                        system,
                        context.time,
                        plus,
                        points,
                        normal,
                        0,
                        context.user_args,
                    )
                    if trace.direct_normal_flux is not None:
                        normal_flux = trace.direct_normal_flux
                    else:
                        if trace.exterior_state is None:
                            raise RuntimeError(
                                "Boundary trace supplied neither state nor normal flux."
                            )
                        normal_flux = method.interface_flux.normal_face_flux(
                            system,
                            plus,
                            trace.exterior_state,
                            normal,
                            context.user_args,
                        ).normal_flux
                    if method.entropy_stability is None:
                        return normal_flux
                    return normal_flux - system.physical_normal_flux(
                        plus, normal, context.user_args
                    )

                return kernel

            actions.append(
                ExteriorFacetAction(
                    field_name,
                    (field_name,),
                    boundary_kernel(patch.boundary),
                    domain=patch.domain,
                    rules=tuple(exterior_rules.items()),
                    action_id=canonical_fingerprint(
                        {"kind": "nodal-dg-boundary", "patch": patch.patch_id}
                    ),
                )
            )
        if source is not None:

            def source_kernel(
                values,
                gradients,
                points,
                physical_weights,
                test_basis,
                test_gradients,
                context,
            ):
                del gradients, test_gradients
                source_values = jnp.asarray(
                    source(context.time, values[0], points, context.user_args)
                )
                if source_values.shape != values[0].shape:
                    raise ValueError("Nodal DG source must match quadrature state shape.")
                return -ein.contract(
                    "cq,cqv,qi->civ",
                    physical_weights,
                    source_values,
                    test_basis,
                    backend="jax",
                )

            actions.append(
                CellResidualAction(
                    field_name,
                    (field_name,),
                    source_kernel,
                    domain=discretization.cell_domain,
                    rules=tuple(volume_rules.items()),
                    action_id="nodal-dg-source",
                )
            )
        form = FiniteElementForm("nodal-dg-conservation", field_name, tuple(actions))
        compiled = CompiledFiniteElementProblem(
            form,
            discretization,
            execution_policy=FiniteElementExecutionPolicy(
                realization="matrix_free",
                local_kernel="dense",
                accumulation=method.accumulation,
            ),
        )
        mass_inverse = PreparedDiscontinuousMassInverse(
            discretization, field_name, volume_rules
        )
        volume_degree = max(volume_degrees.values())
        interior_degree = max(interior_degrees.values())
        exterior_degree = max(exterior_degrees.values())
        volume_evidence = QuadratureEvidence(
            "volume",
            method.volume_quadrature,
            volume_degree,
            exact=False,
            aliasing_status="heuristic-overintegration-nonpolynomial-flux",
        )
        interior_evidence = QuadratureEvidence(
            "interior-facet",
            method.interior_facet_quadrature,
            interior_degree,
            exact=False,
            aliasing_status="heuristic-overintegration-nonpolynomial-flux",
        )
        exterior_evidence = QuadratureEvidence(
            "exterior-facet",
            method.exterior_facet_quadrature,
            exterior_degree,
            exact=False,
            aliasing_status="heuristic-overintegration-nonpolynomial-flux",
        )
        report = NodalDGPreparationReport(
            volume_evidence,
            interior_evidence,
            exterior_evidence,
            mass_inverse.evidence.evidence_id,
            compiled.compilation_id,
        )
        self.system = system
        self.discretization = discretization
        self.method = method
        self.boundaries = boundaries
        self.entropy_pair = entropy_pair
        self.source = source
        self.compiled_finite_element_problem = compiled
        self.mass_inverse = mass_inverse
        self.mortar_routes = mortar_routes
        self.mortar_batches = mortar_batches
        self.periodic_routes = periodic_routes
        self.hybrid_boundary_routes = hybrid_boundary_routes
        self.boundary_batches = boundary_batches
        self.three_dimensional_interface_routes = three_dimensional_interface_routes
        self.geometry_quality = geometry_quality
        self.entropy_operators = entropy_operators
        self.report = report
        self.dynamics_id = canonical_fingerprint(
            {
                "kind": "prepared-nodal-dg-conservation",
                "system": system.system_id,
                "discretization": discretization.prepared_id,
                "method": method.method_id,
                "boundaries": boundaries.boundary_set_id,
                "mortars": tuple(route.route_id for route in mortar_routes),
                "mortar_batches": tuple(batch.batch_id for batch in mortar_batches),
                "periodic_routes": tuple(route.route_id for route in periodic_routes),
                "hybrid_boundaries": tuple(
                    route.route_id for route in hybrid_boundary_routes
                ),
                "boundary_batches": tuple(batch.batch_id for batch in boundary_batches),
                "three_dimensional_interfaces": tuple(
                    route.route_id for route in three_dimensional_interface_routes
                ),
                "entropy_operators": tuple(
                    None if value is None else value.operator_id
                    for value in entropy_operators
                ),
                "geometry_quality": geometry_quality.evidence_id,
                "report": report.report_id,
            }
        )

    @property
    def state_space(self):
        return self.compiled_finite_element_problem.state_space

    def _state(self, state: ArrayLike, /) -> Array:
        return self.state_space.validate(jnp.asarray(state))

    def _context(self, time: Array, args: Any, /) -> FiniteElementExecutionContext:
        if isinstance(args, FiniteElementExecutionContext):
            return FiniteElementExecutionContext(
                args.runtime,
                time=time,
                lift=args.lift,
                lift_rate=args.lift_rate,
                lift_acceleration=args.lift_acceleration,
                user_args=args.user_args,
            )
        return FiniteElementExecutionContext(
            self.discretization.default_runtime,
            time=time,
            user_args=args,
        )

    def _three_dimensional_interface_residual(
        self, state: Array, context: FiniteElementExecutionContext, /
    ) -> Array:
        residual = jnp.zeros_like(state)
        for route in self.three_dimensional_interface_routes:
            plus = ein.contract(
                "qi,iv->qv",
                route.owner_basis,
                state[route.owner_dofs],
                backend="jax",
            )
            minus = ein.contract(
                "qi,iv->qv",
                route.neighbour_basis,
                state[route.neighbour_dofs],
                backend="jax",
            )
            numerical = self.method.interface_flux.normal_face_flux(
                self.system,
                plus,
                minus,
                route.normal,
                context.user_args,
            ).normal_flux
            if self.method.entropy_stability is None:
                owner_flux = numerical
                neighbour_flux = -numerical
            else:
                owner_flux = numerical - self.system.physical_normal_flux(
                    plus, route.normal, context.user_args
                )
                neighbour_flux = -numerical + self.system.physical_normal_flux(
                    minus, route.normal, context.user_args
                )
            owner = ein.contract(
                "q,qi,qv->iv",
                route.physical_weights,
                route.owner_basis,
                owner_flux,
                backend="jax",
            )
            neighbour = ein.contract(
                "q,qi,qv->iv",
                route.physical_weights,
                route.neighbour_basis,
                neighbour_flux,
                backend="jax",
            )
            residual = residual.at[route.owner_dofs].add(owner)
            residual = residual.at[route.neighbour_dofs].add(neighbour)
        return residual

    def _entropy_volume_residual(
        self, state: Array, context: FiniteElementExecutionContext, /
    ) -> Array:
        if self.method.entropy_stability is None:
            return jnp.zeros_like(state)
        residual = jnp.zeros_like(state)
        for block_index, (block, element, operator) in enumerate(
            zip(
                self.discretization.mesh.blocks,
                self.discretization.elements[0],
                self.entropy_operators,
                strict=True,
            )
        ):
            if operator is None:
                raise ValueError(
                    "Entropy flux differencing is unavailable for this cell kind."
                )
            routes = self.discretization.dof_maps[0].cell_dofs[block_index]
            local = state[routes]
            coordinate_element = self.discretization.coordinate_elements[block_index]
            coordinate_basis, coordinate_gradients = coordinate_element.tabulate(
                element.reference_nodes
            )
            coordinate_routes = self.discretization.coordinate_dofs[block_index]
            metric = FiniteElementMetricData(
                coordinate_basis,
                coordinate_gradients,
                context.runtime.coordinates[coordinate_routes],
                jnp.ones((element.local_dof_count,), dtype=state.real.dtype),
            )
            local_residual = operator.flux_differencing_dual(
                self.system,
                local,
                metric.cofactor,
                self.method.entropy_stability,
                context.user_args,
            )
            residual = residual.at[routes].set(local_residual)
        return residual

    def _viscous_weak_residual(
        self, state: Array, context: FiniteElementExecutionContext, /
    ) -> Array:
        if self.method.viscous is None:
            return jnp.zeros_like(state)
        dimension = self.system.dimension
        gradient = jnp.zeros(state.shape + (dimension,), dtype=state.dtype)
        correction_dual = jnp.zeros_like(gradient)
        for block_index, element in enumerate(self.discretization.elements[0]):
            routes = self.discretization.dof_maps[0].cell_dofs[block_index]
            coordinate_element = self.discretization.coordinate_elements[block_index]
            coordinate_basis, coordinate_gradients = coordinate_element.tabulate(
                element.reference_nodes
            )
            coordinate_routes = self.discretization.coordinate_dofs[block_index]
            metric = FiniteElementMetricData(
                coordinate_basis,
                coordinate_gradients,
                context.runtime.coordinates[coordinate_routes],
                jnp.ones((element.local_dof_count,), dtype=state.dtype),
            )
            raw = ein.contract(
                "cqid,civ->cqvd",
                metric.physical_gradients(element.tabulate(element.reference_nodes)[1]),
                state[routes],
                backend="jax",
            )
            gradient = gradient.at[routes].set(raw)

        beta = self.method.viscous.beta
        for route in self.mortar_routes:
            plus = route.mortar.interpolate_left(state[route.owner_dofs])
            minus = route.mortar.interpolate_right(state[route.neighbour_dofs])
            common = 0.5 * (plus + minus) + beta * (plus - minus)
            owner_vector = (common - plus)[..., :, None] * route.normal[..., None, :]
            neighbour_vector = (common - minus)[..., :, None] * (
                -route.normal[..., None, :]
            )
            owner = ein.contract(
                "iq,q,qvd->ivd",
                route.mortar.left_raw_dual_pullback,
                route.mortar.physical_weights,
                owner_vector,
                backend="jax",
            )
            neighbour = ein.contract(
                "iq,q,qvd->ivd",
                route.mortar.right_raw_dual_pullback,
                route.mortar.physical_weights,
                neighbour_vector,
                backend="jax",
            )
            correction_dual = correction_dual.at[route.owner_dofs].add(owner)
            correction_dual = correction_dual.at[route.neighbour_dofs].add(neighbour)
        for route in self.periodic_routes:
            plus = ein.contract(
                "qi,iv->qv",
                route.owner_basis,
                state[route.owner_dofs],
                backend="jax",
            )
            neighbour = ein.contract(
                "qi,iv->qv",
                route.neighbour_basis,
                state[route.neighbour_dofs],
                backend="jax",
            )
            minus = ein.contract(
                "ji,qj->qi",
                route.component_transform,
                neighbour,
                backend="jax",
            )
            common = 0.5 * (plus + minus) + beta * (plus - minus)
            owner_vector = (common - plus)[..., :, None] * route.normal[..., None, :]
            neighbour_owner = (common - minus)[..., :, None] * (
                -route.normal[..., None, :]
            )
            neighbour_vector = ein.contract(
                "ab,qbd,ed->qae",
                route.component_transform,
                neighbour_owner,
                route.coordinate_transform,
                backend="jax",
            )
            owner = ein.contract(
                "q,qi,qvd->ivd",
                route.physical_weights,
                route.owner_basis,
                owner_vector,
                backend="jax",
            )
            neighbour_correction = ein.contract(
                "q,qi,qvd->ivd",
                route.physical_weights,
                route.neighbour_basis,
                neighbour_vector,
                backend="jax",
            )
            correction_dual = correction_dual.at[route.owner_dofs].add(owner)
            correction_dual = correction_dual.at[route.neighbour_dofs].add(
                neighbour_correction
            )
        for route in self.three_dimensional_interface_routes:
            plus = ein.contract(
                "qi,iv->qv",
                route.owner_basis,
                state[route.owner_dofs],
                backend="jax",
            )
            minus = ein.contract(
                "qi,iv->qv",
                route.neighbour_basis,
                state[route.neighbour_dofs],
                backend="jax",
            )
            common = 0.5 * (plus + minus) + beta * (plus - minus)
            owner_vector = (common - plus)[..., :, None] * route.normal[..., None, :]
            neighbour_vector = (common - minus)[..., :, None] * (
                -route.normal[..., None, :]
            )
            owner = ein.contract(
                "q,qi,qvd->ivd",
                route.physical_weights,
                route.owner_basis,
                owner_vector,
                backend="jax",
            )
            neighbour = ein.contract(
                "q,qi,qvd->ivd",
                route.physical_weights,
                route.neighbour_basis,
                neighbour_vector,
                backend="jax",
            )
            correction_dual = correction_dual.at[route.owner_dofs].add(owner)
            correction_dual = correction_dual.at[route.neighbour_dofs].add(neighbour)
        for route in self.hybrid_boundary_routes:
            if isinstance(route.boundary, PrescribedNormalFluxBoundary):
                raise ValueError(
                    "Viscous DG requires boundary state and gradient traces."
                )
            plus = ein.contract(
                "qi,iv->qv",
                route.owner_basis,
                state[route.owner_dofs],
                backend="jax",
            )
            trace = evaluate_conservation_boundary(
                route.boundary,
                self.system,
                context.time,
                plus,
                route.physical_points,
                route.normal,
                0,
                context.user_args,
            )
            if trace.viscous_state_trace is None:
                raise ValueError("Boundary supplied no viscous state trace.")
            common = 0.5 * (plus + trace.viscous_state_trace) + beta * (
                plus - trace.viscous_state_trace
            )
            vector = (common - plus)[..., :, None] * route.normal[..., None, :]
            local = ein.contract(
                "q,qi,qvd->ivd",
                route.physical_weights,
                route.owner_basis,
                vector,
                backend="jax",
            )
            correction_dual = correction_dual.at[route.owner_dofs].add(local)

        gradient = gradient + self.mass_inverse.apply(correction_dual)
        residual = jnp.zeros_like(state)
        for block_index, (block, element) in enumerate(
            zip(
                self.discretization.mesh.blocks,
                self.discretization.elements[0],
                strict=True,
            )
        ):
            routes = self.discretization.dof_maps[0].cell_dofs[block_index]
            volume_rule, _facet_rule = _rules(
                block.cell_kind,
                self.report.volume_quadrature.selected_degree,
                self.report.exterior_quadrature.selected_degree,
            )
            from ...integration._rules import reference_rule_data

            data = reference_rule_data(volume_rule)
            geometry = self.discretization.evaluate_block_geometry(
                self.discretization.field_spaces[0].name,
                block_index,
                context.runtime.coordinates,
                data.points,
                data.weights,
            )
            values = ein.contract(
                "qi,civ->cqv",
                geometry.basis_values,
                state[routes],
                backend="jax",
            )
            gradients = ein.contract(
                "qi,civd->cqvd",
                geometry.basis_values,
                gradient[routes],
                backend="jax",
            )
            flux = self.system.viscous_flux(values, gradients, context.user_args)
            local = ein.contract(
                "cq,cqid,cqvd->civ",
                geometry.physical_weights,
                geometry.physical_gradients,
                flux,
                backend="jax",
            )
            residual = residual.at[routes].set(local)

        def viscous_common(
            plus,
            minus,
            plus_gradient,
            minus_gradient,
            normal,
        ):
            plus_flux = self.system.viscous_normal_flux(
                plus, plus_gradient, normal, context.user_args
            )
            minus_flux = self.system.viscous_normal_flux(
                minus, minus_gradient, normal, context.user_args
            )
            return (
                0.5 * (plus_flux + minus_flux)
                + beta * (plus_flux - minus_flux)
                + self.method.viscous.penalty * (minus - plus)
            )

        for route in self.mortar_routes:
            plus = route.mortar.interpolate_left(state[route.owner_dofs])
            minus = route.mortar.interpolate_right(state[route.neighbour_dofs])
            plus_gradient = route.mortar.interpolate_left(gradient[route.owner_dofs])
            minus_gradient = route.mortar.interpolate_right(
                gradient[route.neighbour_dofs]
            )
            common = viscous_common(
                plus, minus, plus_gradient, minus_gradient, route.normal
            )
            owner = ein.contract(
                "iq,q,qv->iv",
                route.mortar.left_raw_dual_pullback,
                route.mortar.physical_weights,
                -common,
                backend="jax",
            )
            neighbour = ein.contract(
                "iq,q,qv->iv",
                route.mortar.right_raw_dual_pullback,
                route.mortar.physical_weights,
                common,
                backend="jax",
            )
            residual = residual.at[route.owner_dofs].add(owner)
            residual = residual.at[route.neighbour_dofs].add(neighbour)
        for route in self.three_dimensional_interface_routes:
            plus = ein.contract(
                "qi,iv->qv",
                route.owner_basis,
                state[route.owner_dofs],
                backend="jax",
            )
            minus = ein.contract(
                "qi,iv->qv",
                route.neighbour_basis,
                state[route.neighbour_dofs],
                backend="jax",
            )
            plus_gradient = ein.contract(
                "qi,ivd->qvd",
                route.owner_basis,
                gradient[route.owner_dofs],
                backend="jax",
            )
            minus_gradient = ein.contract(
                "qi,ivd->qvd",
                route.neighbour_basis,
                gradient[route.neighbour_dofs],
                backend="jax",
            )
            common = viscous_common(
                plus, minus, plus_gradient, minus_gradient, route.normal
            )
            owner = -ein.contract(
                "q,qi,qv->iv",
                route.physical_weights,
                route.owner_basis,
                common,
                backend="jax",
            )
            neighbour = ein.contract(
                "q,qi,qv->iv",
                route.physical_weights,
                route.neighbour_basis,
                common,
                backend="jax",
            )
            residual = residual.at[route.owner_dofs].add(owner)
            residual = residual.at[route.neighbour_dofs].add(neighbour)
        for route in self.hybrid_boundary_routes:
            plus = ein.contract(
                "qi,iv->qv",
                route.owner_basis,
                state[route.owner_dofs],
                backend="jax",
            )
            plus_gradient = ein.contract(
                "qi,ivd->qvd",
                route.owner_basis,
                gradient[route.owner_dofs],
                backend="jax",
            )
            trace = evaluate_conservation_boundary(
                route.boundary,
                self.system,
                context.time,
                plus,
                route.physical_points,
                route.normal,
                0,
                context.user_args,
            )
            if trace.viscous_state_trace is None:
                raise ValueError("Boundary supplied no viscous state trace.")
            closure = self.method.viscous.boundary_closure(route.boundary.boundary_id)
            minus_gradient = closure.gradient_trace(
                context.time,
                plus,
                plus_gradient,
                route.physical_points,
                route.normal,
                context.user_args,
            )
            default_common = viscous_common(
                plus,
                trace.viscous_state_trace,
                plus_gradient,
                minus_gradient,
                route.normal,
            )
            common = closure.normal_flux(
                context.time,
                plus,
                trace.viscous_state_trace,
                plus_gradient,
                minus_gradient,
                route.normal,
                default_common,
                context.user_args,
            )
            local = -ein.contract(
                "q,qi,qv->iv",
                route.physical_weights,
                route.owner_basis,
                common,
                backend="jax",
            )
            residual = residual.at[route.owner_dofs].add(local)
        return residual

    def _mortar_residual(
        self, state: Array, context: FiniteElementExecutionContext, /
    ) -> Array:
        residual = jnp.zeros_like(state)
        for batch in self.mortar_batches:
            plus = ein.contract(
                "rqi,riv->rqv",
                batch.left_interpolation,
                state[batch.owner_dofs],
                backend="jax",
            )
            minus = ein.contract(
                "rqi,riv->rqv",
                batch.right_interpolation,
                state[batch.neighbour_dofs],
                backend="jax",
            )
            flux = self.method.interface_flux.normal_face_flux(
                self.system,
                plus,
                minus,
                batch.normal,
                context.user_args,
            ).normal_flux
            if self.method.entropy_stability is None:
                owner_flux = flux
                neighbour_flux = -flux
            else:
                owner_flux = flux - self.system.physical_normal_flux(
                    plus, batch.normal, context.user_args
                )
                neighbour_flux = -flux + self.system.physical_normal_flux(
                    minus, batch.normal, context.user_args
                )
            owner = ein.contract(
                "riq,rq,rqv->riv",
                batch.left_dual_pullback,
                batch.physical_weights,
                owner_flux,
                backend="jax",
            )
            neighbour = ein.contract(
                "riq,rq,rqv->riv",
                batch.right_dual_pullback,
                batch.physical_weights,
                neighbour_flux,
                backend="jax",
            )
            residual = residual.at[batch.owner_dofs].add(owner)
            residual = residual.at[batch.neighbour_dofs].add(neighbour)
        return residual

    def _periodic_residual(
        self, state: Array, context: FiniteElementExecutionContext, /
    ) -> Array:
        residual = jnp.zeros_like(state)
        for route in self.periodic_routes:
            owner_state = ein.contract(
                "qi,iv->qv",
                route.owner_basis,
                state[route.owner_dofs],
                backend="jax",
            )
            neighbour_state = ein.contract(
                "qi,iv->qv",
                route.neighbour_basis,
                state[route.neighbour_dofs],
                backend="jax",
            )
            neighbour_in_owner_frame = ein.contract(
                "ji,qj->qi",
                route.component_transform,
                neighbour_state,
                backend="jax",
            )
            if self.method.entropy_stability is None:
                owner_flux = self.method.interface_flux.normal_face_flux(
                    self.system,
                    owner_state,
                    neighbour_in_owner_frame,
                    route.normal,
                    context.user_args,
                ).normal_flux
                neighbour_flux = -ein.contract(
                    "ij,qj->qi",
                    route.component_transform,
                    owner_flux,
                    backend="jax",
                )
            else:
                numerical = self.method.interface_flux.normal_face_flux(
                    self.system,
                    owner_state,
                    neighbour_in_owner_frame,
                    route.normal,
                    context.user_args,
                ).normal_flux
                owner_flux = numerical - self.system.physical_normal_flux(
                    owner_state, route.normal, context.user_args
                )
                neighbour_correction = -numerical + self.system.physical_normal_flux(
                    neighbour_in_owner_frame,
                    route.normal,
                    context.user_args,
                )
                neighbour_flux = ein.contract(
                    "ij,qj->qi",
                    route.component_transform,
                    neighbour_correction,
                    backend="jax",
                )
            owner = ein.contract(
                "q,qi,qv->iv",
                route.physical_weights,
                route.owner_basis,
                owner_flux,
                backend="jax",
            )
            neighbour = ein.contract(
                "q,qi,qv->iv",
                route.physical_weights,
                route.neighbour_basis,
                neighbour_flux,
                backend="jax",
            )
            residual = residual.at[route.owner_dofs].add(owner)
            residual = residual.at[route.neighbour_dofs].add(neighbour)
        return residual

    def _hybrid_boundary_residual(
        self, state: Array, context: FiniteElementExecutionContext, /
    ) -> Array:
        residual = jnp.zeros_like(state)
        for batch in self.boundary_batches:
            plus = ein.contract(
                "rqi,riv->rqv",
                batch.owner_basis,
                state[batch.owner_dofs],
                backend="jax",
            )
            trace = evaluate_conservation_boundary(
                batch.boundary,
                self.system,
                context.time,
                plus,
                batch.physical_points,
                batch.normal,
                0,
                context.user_args,
            )
            if trace.direct_normal_flux is not None:
                flux = trace.direct_normal_flux
            else:
                if trace.exterior_state is None:
                    raise RuntimeError(
                        "Boundary trace supplied neither state nor normal flux."
                    )
                flux = self.method.interface_flux.normal_face_flux(
                    self.system,
                    plus,
                    trace.exterior_state,
                    batch.normal,
                    context.user_args,
                ).normal_flux
            if self.method.entropy_stability is not None:
                flux = flux - self.system.physical_normal_flux(
                    plus, batch.normal, context.user_args
                )
            local = ein.contract(
                "rq,rqi,rqv->riv",
                batch.physical_weights,
                batch.owner_basis,
                flux,
                backend="jax",
            )
            residual = residual.at[batch.owner_dofs].add(local)
        return residual

    def weak_residual(self, time: Array, state: ArrayLike, args: Any = None, /) -> Array:
        value = self._state(state)
        context = self._context(jnp.asarray(time), args)
        residual = self.compiled_finite_element_problem.residual(value, context)
        return (
            residual
            + self._mortar_residual(value, context)
            + self._periodic_residual(value, context)
            + self._three_dimensional_interface_residual(value, context)
            + self._hybrid_boundary_residual(value, context)
            + self._entropy_volume_residual(value, context)
            + self._viscous_weak_residual(value, context)
        )

    def _uncorrected_mass_inverted_rate(
        self, time: Array, state: ArrayLike, args: Any = None, /
    ) -> Array:
        residual = self.weak_residual(time, state, args)
        return -self.mass_inverse.apply(residual)

    def mass_inverted_rate(
        self, time: Array, state: ArrayLike, args: Any = None, /
    ) -> Array:
        value = self._state(state)
        raw = self._uncorrected_mass_inverted_rate(time, value, args)
        if self.method.equilibrium is None:
            return raw
        coordinates = self.discretization.dof_maps[0].dof_coordinates
        equilibrium = self.method.equilibrium.state(
            jnp.asarray(time),
            coordinates,
            args,
            self.system.component_count,
        )
        equilibrium_rate = self._uncorrected_mass_inverted_rate(time, equilibrium, args)
        derivative = self.method.equilibrium.time_derivative(
            jnp.asarray(time), equilibrium, coordinates, args
        )
        return raw - equilibrium_rate + derivative

    def __call__(self, time: Array, state: ArrayLike, args: Any = None) -> Array:
        return self.mass_inverted_rate(time, state, args)

    def face_fluxes(
        self, time: Array, state: ArrayLike, args: Any = None, /
    ) -> NodalDGFaceFluxes:
        value = self._state(state)
        context = self._context(jnp.asarray(time), args)
        fluxes = []
        speeds = []
        weights = []
        integrated = []
        entropy_productions = []
        kinds = []
        identifiers = []

        def interface_entropy(left, right, normal_flux, normal):
            if self.entropy_pair is None:
                return jnp.zeros(normal_flux.shape[:-1], dtype=normal_flux.dtype)
            return entropy_mortar_evidence(
                self.entropy_pair,
                left,
                right,
                normal_flux,
                normal,
                tolerance=(
                    self.method.entropy_stability.tolerance
                    if self.method.entropy_stability is not None
                    else 1.0e-10
                ),
            ).entropy_production

        for route in self.mortar_routes:
            plus = route.mortar.interpolate_left(value[route.owner_dofs])
            minus = route.mortar.interpolate_right(value[route.neighbour_dofs])
            result = self.method.interface_flux.normal_face_flux(
                self.system, plus, minus, route.normal, context.user_args
            )
            physical_weights = route.mortar.physical_weights
            fluxes.append(result.normal_flux)
            speeds.append(result.max_speed)
            weights.append(physical_weights)
            integrated.append(
                ein.contract(
                    "q,qv->v",
                    physical_weights,
                    result.normal_flux,
                    backend="jax",
                )
            )
            entropy_productions.append(
                interface_entropy(plus, minus, result.normal_flux, route.normal)
            )
            kinds.append("mortar")
            identifiers.append(route.route_id)
        for route in self.periodic_routes:
            plus = ein.contract(
                "qi,iv->qv",
                route.owner_basis,
                value[route.owner_dofs],
                backend="jax",
            )
            neighbour = ein.contract(
                "qi,iv->qv",
                route.neighbour_basis,
                value[route.neighbour_dofs],
                backend="jax",
            )
            minus = ein.contract(
                "ji,qj->qi",
                route.component_transform,
                neighbour,
                backend="jax",
            )
            result = self.method.interface_flux.normal_face_flux(
                self.system, plus, minus, route.normal, context.user_args
            )
            fluxes.append(result.normal_flux)
            speeds.append(result.max_speed)
            weights.append(route.physical_weights)
            integrated.append(
                ein.contract(
                    "q,qv->v",
                    route.physical_weights,
                    result.normal_flux,
                    backend="jax",
                )
            )
            entropy_productions.append(
                interface_entropy(plus, minus, result.normal_flux, route.normal)
            )
            kinds.append("periodic")
            identifiers.append(route.route_id)
        for route in self.three_dimensional_interface_routes:
            plus = ein.contract(
                "qi,iv->qv",
                route.owner_basis,
                value[route.owner_dofs],
                backend="jax",
            )
            minus = ein.contract(
                "qi,iv->qv",
                route.neighbour_basis,
                value[route.neighbour_dofs],
                backend="jax",
            )
            result = self.method.interface_flux.normal_face_flux(
                self.system, plus, minus, route.normal, context.user_args
            )
            fluxes.append(result.normal_flux)
            speeds.append(result.max_speed)
            weights.append(route.physical_weights)
            integrated.append(
                ein.contract(
                    "q,qv->v",
                    route.physical_weights,
                    result.normal_flux,
                    backend="jax",
                )
            )
            entropy_productions.append(
                interface_entropy(plus, minus, result.normal_flux, route.normal)
            )
            kinds.append("conforming")
            identifiers.append(route.route_id)
        for route in self.hybrid_boundary_routes:
            plus = ein.contract(
                "qi,iv->qv",
                route.owner_basis,
                value[route.owner_dofs],
                backend="jax",
            )
            trace = evaluate_conservation_boundary(
                route.boundary,
                self.system,
                context.time,
                plus,
                route.physical_points,
                route.normal,
                0,
                context.user_args,
            )
            if trace.direct_normal_flux is not None:
                normal_flux = trace.direct_normal_flux
                signal_speed = self.system.max_normal_wave_speed(
                    plus, plus, route.normal, context.user_args
                )
            else:
                if trace.exterior_state is None:
                    raise RuntimeError(
                        "Boundary trace supplied neither state nor normal flux."
                    )
                result = self.method.interface_flux.normal_face_flux(
                    self.system,
                    plus,
                    trace.exterior_state,
                    route.normal,
                    context.user_args,
                )
                normal_flux = result.normal_flux
                signal_speed = result.max_speed
            fluxes.append(normal_flux)
            speeds.append(signal_speed)
            weights.append(route.physical_weights)
            integrated.append(
                ein.contract(
                    "q,qv->v",
                    route.physical_weights,
                    normal_flux,
                    backend="jax",
                )
            )
            if self.entropy_pair is None or self.method.entropy_stability is None:
                entropy_productions.append(
                    jnp.zeros(normal_flux.shape[:-1], dtype=normal_flux.dtype)
                )
            else:
                variables = self.entropy_pair.entropy_variables(plus)
                potential = sum(
                    route.normal[..., direction]
                    * self.entropy_pair.entropy_potential(plus, direction)
                    for direction in range(route.normal.shape[-1])
                )
                numerical_entropy = (
                    ein.contract("qv,qv->q", variables, normal_flux, backend="jax")
                    - potential
                )
                contract = self.method.entropy_stability.boundary_contract(
                    route.boundary.boundary_id
                )
                allowed = contract.allowed_supply(
                    context.time,
                    plus,
                    route.physical_points,
                    route.normal,
                    numerical_entropy,
                    self.entropy_pair,
                    context.user_args,
                )
                evidence = boundary_entropy_evidence(
                    self.entropy_pair,
                    plus,
                    normal_flux,
                    route.normal,
                    allowed,
                    tolerance=self.method.entropy_stability.tolerance,
                )
                entropy_productions.append(evidence.defect)
            kinds.append("boundary")
            identifiers.append(route.route_id)
        return NodalDGFaceFluxes(
            tuple(fluxes),
            tuple(speeds),
            tuple(weights),
            tuple(integrated),
            tuple(entropy_productions),
            tuple(kinds),
            tuple(identifiers),
        )

    def stable_step_evidence(
        self, state: ArrayLike, args: Any = None, /, *, cfl: float = 0.45
    ) -> NodalDGStableStepEvidence:
        value = self._state(state)
        cfl_ = float(cfl)
        if not math.isfinite(cfl_) or cfl_ <= 0.0:
            raise ValueError("Nodal DG CFL must be finite and positive.")
        maximum_speed = jnp.max(
            jnp.stack(
                tuple(
                    jnp.max(self.system.max_wave_speed(value, value, axis, args))
                    for axis in range(self.system.dimension)
                )
            )
        )
        measures = self.discretization.measures[0].weights
        minimum_length = jnp.min(
            measures ** (1.0 / self.discretization.mesh.topological_dimension)
        )
        degree = max(
            element.degree
            for elements in self.discretization.elements
            for element in elements
        )
        advective_rate = (2 * degree + 1) * maximum_speed / minimum_length
        diffusive_rate = jnp.zeros((), dtype=value.dtype)
        if self.method.viscous is not None:
            temperature = self.system.temperature(value)
            properties = self.system.transport.properties(temperature, value, args)
            density = value[..., 0]
            heat_capacity = self.system.material.gas_constant / (self.system.gamma - 1.0)
            diffusivity = jnp.maximum(
                properties.dynamic_viscosity / density,
                properties.thermal_conductivity / (density * heat_capacity),
            )
            diffusive_rate = (degree + 1) ** 2 * jnp.max(diffusivity) / minimum_length**2
        maximum_rate = jnp.maximum(advective_rate, diffusive_rate)
        step = jnp.asarray(cfl_, dtype=value.dtype) / jnp.where(
            maximum_rate > 0.0, maximum_rate, jnp.inf
        )
        return NodalDGStableStepEvidence(
            step,
            advective_rate,
            diffusive_rate,
            minimum_length,
            degree,
            cfl_,
            jnp.isfinite(step) & (step > 0.0),
            self.method.method_id,
        )

    def stable_step(
        self, state: ArrayLike, args: Any = None, /, *, cfl: float = 0.45
    ) -> Array:
        return self.stable_step_evidence(state, args, cfl=cfl).step

    def residual_with_diagnostics(
        self, time: Array, state: ArrayLike, args: Any = None, /
    ) -> tuple[Array, NodalDGConservationDiagnostics]:
        value = self._state(state)
        rate = self(time, value, args)
        total_integral = jnp.zeros((self.system.component_count,), dtype=value.dtype)
        conservation_rate = jnp.zeros_like(total_integral)
        for routes, matrices in zip(
            self.mass_inverse.routes,
            self.mass_inverse.mass_matrices,
            strict=True,
        ):
            local_value = value[routes]
            local_rate = rate[routes]
            weighted_value = ein.contract(
                "cij,cjv->civ", matrices, local_value, backend="jax"
            )
            weighted_rate = ein.contract(
                "cij,cjv->civ", matrices, local_rate, backend="jax"
            )
            total_integral = total_integral + jnp.sum(weighted_value, axis=(0, 1))
            conservation_rate = conservation_rate + jnp.sum(weighted_rate, axis=(0, 1))
        faces = self.face_fluxes(time, value, args)
        interface_entropy_production = (
            None
            if self.entropy_pair is None
            else sum(
                (
                    jnp.sum(weight * production)
                    for kind, weight, production in zip(
                        faces.route_kinds,
                        faces.physical_weights,
                        faces.entropy_productions,
                        strict=True,
                    )
                    if kind != "boundary"
                ),
                jnp.asarray(0.0, dtype=value.dtype),
            )
        )
        boundary_entropy_defect = (
            None
            if self.entropy_pair is None
            else sum(
                (
                    jnp.sum(weight * production)
                    for kind, weight, production in zip(
                        faces.route_kinds,
                        faces.physical_weights,
                        faces.entropy_productions,
                        strict=True,
                    )
                    if kind == "boundary"
                ),
                jnp.asarray(0.0, dtype=value.dtype),
            )
        )
        boundary_flux_rate = sum(
            (
                flux
                for kind, flux in zip(
                    faces.route_kinds, faces.integrated_fluxes, strict=True
                )
                if kind == "boundary"
            ),
            jnp.zeros_like(total_integral),
        )
        source_integral = jnp.zeros_like(total_integral)
        if self.source is not None:
            context = self._context(jnp.asarray(time), args)
            coordinates = self.discretization.dof_maps[0].dof_coordinates
            source_values = jnp.asarray(
                self.source(context.time, value, coordinates, context.user_args)
            )
            if source_values.shape != value.shape:
                raise ValueError("Nodal DG source must match the global state shape.")
            for routes, matrices in zip(
                self.mass_inverse.routes,
                self.mass_inverse.mass_matrices,
                strict=True,
            ):
                weighted_source = ein.contract(
                    "cij,cjv->civ",
                    matrices,
                    source_values[routes],
                    backend="jax",
                )
                source_integral = source_integral + jnp.sum(weighted_source, axis=(0, 1))
        balance_defect = conservation_rate + boundary_flux_rate - source_integral
        admissible = (
            None
            if self.entropy_pair is None
            else jnp.all(self.entropy_pair.admissible(value))
        )
        diagnostics = NodalDGConservationDiagnostics(
            total_integral,
            conservation_rate,
            boundary_flux_rate,
            source_integral,
            balance_defect,
            interface_entropy_production,
            boundary_entropy_defect,
            admissible,
            self.method.method_id,
        )
        return rate, diagnostics

    def viscous_linearize(self, time: Array, state: ArrayLike, args: Any = None, /):
        if self.method.viscous is None:
            raise ValueError("Nodal DG has no viscous operator to linearize.")
        value = self._state(state)
        context = self._context(jnp.asarray(time), args)

        def viscous_rate(candidate):
            weak = self._viscous_weak_residual(candidate, context)
            return -self.mass_inverse.apply(weak)

        residual, pushforward = jax.linearize(viscous_rate, value)
        _, pullback = jax.vjp(viscous_rate, value)
        return residual, pushforward, pullback

    def linearize(self, time: Array, state: ArrayLike, args: Any = None, /):
        value = self._state(state)
        residual, pushforward = jax.linearize(
            lambda candidate: self(time, candidate, args), value
        )
        _, pullback = jax.vjp(lambda candidate: self(time, candidate, args), value)
        return residual, pushforward, pullback


__all__ = [
    "NodalDGConservationDiagnostics",
    "NodalDGFaceFluxes",
    "NodalDGStableStepEvidence",
    "NodalDGConservationMethodPlan",
    "NodalDGPreparationReport",
    "PreparedNodalDGConservationDynamics",
]
