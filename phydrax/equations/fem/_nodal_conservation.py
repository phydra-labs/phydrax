#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization._cell_complex import (
    PolygonalConnectivity,
    PolyhedralConnectivity,
)
from ...discretization._conservation_boundary import (
    AbstractConservationBoundary,
    PrescribedNormalFluxBoundary,
)
from ...discretization.fem._boundary import FiniteElementBoundarySet
from ...discretization.fem._generic import (
    FiniteElementDiscretization,
    IntegrationDomain,
)
from ...discretization.fem._mortar import (
    FiniteElementMortarPlan,
    serial_finite_element_mortar_plan,
)
from ...discretization.fem._reference_operator import _map_face_rule
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
from .._hyperbolic_systems import CompressibleNavierStokesSystem
from ._mass_inverse import PreparedDiscontinuousMassInverse
from ._operators import FiniteElementFacetMetricData, FiniteElementMetricData
from ._quadrature import QuadratureAccuracyPolicy, QuadratureEvidence
from ._viscous_conservation import LDGViscousFluxPlan


class NodalDGConservationMethodPlan(StrictModule, NonTrainableState):
    interface_flux: AbstractArbitraryNormalNumericalFluxPlan
    volume_quadrature: QuadratureAccuracyPolicy
    interior_facet_quadrature: QuadratureAccuracyPolicy
    exterior_facet_quadrature: QuadratureAccuracyPolicy
    viscous: LDGViscousFluxPlan | None
    accumulation: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        interface_flux: AbstractArbitraryNormalNumericalFluxPlan,
        /,
        *,
        volume_quadrature: QuadratureAccuracyPolicy | None = None,
        interior_facet_quadrature: QuadratureAccuracyPolicy | None = None,
        exterior_facet_quadrature: QuadratureAccuracyPolicy | None = None,
        viscous: LDGViscousFluxPlan | None = None,
        accumulation: str = "deterministic",
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
        if viscous is not None and not isinstance(viscous, LDGViscousFluxPlan):
            raise TypeError("viscous must be LDGViscousFluxPlan or None.")
        accumulation_ = str(accumulation)
        if accumulation_ not in ("fast", "deterministic", "compensated"):
            raise ValueError("Unknown nodal DG accumulation policy.")
        self.interface_flux = interface_flux
        self.volume_quadrature = volume
        self.interior_facet_quadrature = interior
        self.exterior_facet_quadrature = exterior
        self.viscous = viscous
        self.accumulation = accumulation_
        self.method_id = canonical_fingerprint(
            {
                "kind": "nodal-dg-conservation-method",
                "interface_flux": interface_flux.flux_id,
                "volume_quadrature": volume.policy_id,
                "interior_quadrature": interior.policy_id,
                "exterior_quadrature": exterior.policy_id,
                "viscous": None if viscous is None else viscous.plan_id,
                "mass": "exact-cell-local",
                "accumulation": accumulation_,
                "entropy_evidence": "uncertified",
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
    admissible: Array | None
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


class NodalDGMortarRoute(StrictModule, NonTrainableState):
    owner_dofs: Array
    neighbour_dofs: Array
    normal: Array
    mortar: FiniteElementMortarPlan
    route_id: str = eqx.field(static=True)


class NodalDGHybridBoundaryRoute(StrictModule, NonTrainableState):
    owner_dofs: Array
    basis_values: Array
    basis_gradients: Array
    physical_points: Array
    physical_weights: Array
    normal: Array
    boundary: AbstractConservationBoundary
    route_id: str = eqx.field(static=True)


class NodalDGThreeDimensionalInterfaceRoute(StrictModule, NonTrainableState):
    owner_dofs: Array
    neighbour_dofs: Array
    owner_basis: Array
    neighbour_basis: Array
    physical_points: Array
    physical_weights: Array
    normal: Array
    route_id: str = eqx.field(static=True)


def _same_block_interior_domain(
    discretization: FiniteElementDiscretization, /
) -> IntegrationDomain | None:
    if discretization.mesh.topological_dimension == 3 and isinstance(
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


def _mixed_mortar_routes(
    discretization: FiniteElementDiscretization,
    field_index: int,
    facet_degrees: dict[str, int],
    /,
) -> tuple[NodalDGMortarRoute, ...]:
    from ...integration._rules import (
        GaussLegendreRule,
        GaussLobattoLegendreRule,
        ReferenceIntervalRule,
    )

    if (
        len(discretization.mesh.blocks) == 1
        or discretization.mesh.topological_dimension != 2
    ):
        return ()
    connectivity = discretization.mesh.connectivity
    if not isinstance(connectivity, PolygonalConnectivity):
        raise TypeError("Mixed nodal DG currently requires polygonal connectivity.")
    offsets = np.cumsum(
        (0,) + tuple(block.cell_count for block in discretization.mesh.blocks)
    )
    domain = discretization.interior_facet_domain
    routes = []
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
        if owner_block == neighbour_block:
            continue
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
        right_points = (
            1.0 - quadrature_points
            if owner_sign * neighbour_sign < 0.0
            else quadrature_points
        )
        edge_vertices = np.asarray(connectivity.edges)[int(facet)]
        edge_points = np.asarray(discretization.mesh.coordinates)[edge_vertices]
        tangent = edge_points[1] - edge_points[0]
        measure = float(np.sqrt(tangent @ tangent))
        normal = np.asarray((tangent[1], -tangent[0])) / measure
        owner_vertices = discretization.mesh.blocks[owner_block].vertices[owner_cell]
        owner_center = np.mean(
            np.asarray(discretization.mesh.coordinates)[owner_vertices], axis=0
        )
        midpoint = 0.5 * (edge_points[0] + edge_points[1])
        if float(normal @ (midpoint - owner_center)) < 0.0:
            normal = -normal
        mortar = serial_finite_element_mortar_plan(
            owner_nodes[:, None],
            neighbour_nodes[:, None],
            mortar_nodes,
            quadrature_points,
            quadrature_weights * measure,
            left_evaluation_points=quadrature_points,
            right_evaluation_points=right_points,
            declared_reproduction_degree=min(
                owner_element.degree, neighbour_element.degree
            ),
            left_polynomial_coordinates=owner_nodes[:, None],
            right_polynomial_coordinates=(
                1.0 - neighbour_nodes[:, None]
                if owner_sign * neighbour_sign < 0.0
                else neighbour_nodes[:, None]
            ),
            mortar_polynomial_coordinates=mortar_nodes,
            polynomial_evaluation_points=quadrature_points,
            interface_id=canonical_fingerprint(
                {
                    "kind": "mixed-nodal-dg-mortar",
                    "facet": int(facet),
                    "owner_element": owner_element.element_id,
                    "neighbour_element": neighbour_element.element_id,
                }
            ),
        )
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
            NodalDGMortarRoute(
                jnp.asarray(owner_dofs, dtype=jnp.int32),
                jnp.asarray(neighbour_dofs, dtype=jnp.int32),
                jnp.broadcast_to(
                    jnp.asarray(normal),
                    (quadrature_points.shape[0], normal.shape[0]),
                ),
                mortar,
                route_id,
            )
        )
    return tuple(routes)


def _hybrid_boundary_routes(
    discretization: FiniteElementDiscretization,
    field_index: int,
    boundaries: FiniteElementBoundarySet,
    facet_degree: int,
    /,
) -> tuple[NodalDGHybridBoundaryRoute, ...]:
    from ...integration._rules import (
        GaussLegendreRule,
        reference_rule_data,
        ReferenceQuadrilateralRule,
        ReferenceTriangleRule,
    )

    if discretization.mesh.topological_dimension != 3:
        return ()
    if any(
        block.cell_kind not in ("tetrahedron", "prism", "pyramid")
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
        local_cell = int(owner - offsets[block_index])
        block = discretization.mesh.blocks[block_index]
        element = discretization.elements[field_index][block_index]
        coordinate_element = discretization.coordinate_elements[block_index]
        coordinate_routes = discretization.coordinate_dofs[block_index]
        topology = reference_cell_topology(block.cell_kind)
        face_arity = len(topology.entities[2][int(local_facet)])
        rule = (
            ReferenceTriangleRule(axis_rule)
            if face_arity == 3
            else ReferenceQuadrilateralRule(axis_rule)
        )
        data = reference_rule_data(rule)
        points, weights, normals = _map_face_rule(block.cell_kind, int(local_facet), data)
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
            NodalDGHybridBoundaryRoute(
                dof_map.cell_dofs[block_index][local_cell],
                basis_values,
                physical_gradients,
                facet_metric.physical_points[0],
                facet_metric.physical_weights[0],
                facet_metric.normal[0],
                policy_by_facet[int(facet)],
                route_id,
            )
        )
    return tuple(routes)


def _three_dimensional_interface_routes(
    discretization: FiniteElementDiscretization,
    field_index: int,
    facet_degree: int,
    /,
) -> tuple[NodalDGThreeDimensionalInterfaceRoute, ...]:
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
            NodalDGThreeDimensionalInterfaceRoute(
                dof_map.cell_dofs[owner_block][owner_cell],
                dof_map.cell_dofs[neighbour_block][neighbour_cell],
                owner_basis,
                neighbour_basis,
                facet_metric.physical_points[0],
                facet_metric.physical_weights[0],
                facet_metric.normal[0],
                route_id,
            )
        )
    return tuple(routes)


def _rule_count(degree: int, /) -> int:
    return max(1, (int(degree) + 2) // 2)


def _rules(cell_kind: str, volume_degree: int, facet_degree: int, /):
    from ...integration._rules import (
        GaussLegendreRule,
        ReferenceIntervalRule,
        ReferencePrismRule,
        ReferencePyramidRule,
        ReferenceQuadrilateralRule,
        ReferenceTetrahedronRule,
        ReferenceTriangleRule,
    )

    volume_axis = GaussLegendreRule(_rule_count(volume_degree))
    facet_axis = GaussLegendreRule(_rule_count(facet_degree))
    if cell_kind == "triangle":
        return ReferenceTriangleRule(volume_axis), ReferenceIntervalRule(facet_axis)
    if cell_kind == "quadrilateral":
        return ReferenceQuadrilateralRule(volume_axis), ReferenceIntervalRule(facet_axis)
    if cell_kind == "tetrahedron":
        return ReferenceTetrahedronRule(volume_axis), ReferenceTriangleRule(facet_axis)
    if cell_kind == "prism":
        return ReferencePrismRule(volume_axis), None
    if cell_kind == "pyramid":
        return ReferencePyramidRule(volume_axis), None
    raise ValueError(
        "Nodal DG supports triangle, quadrilateral, tetrahedron, prism, and "
        "pyramid cells."
    )


class PreparedNodalDGConservationDynamics(StrictModule):
    system: Any
    discretization: FiniteElementDiscretization
    method: NodalDGConservationMethodPlan
    boundaries: FiniteElementBoundarySet
    entropy_pair: ConvexEntropyPair | None
    source: Callable | None = eqx.field(static=True)
    compiled_finite_element_problem: CompiledFiniteElementProblem
    mass_inverse: PreparedDiscontinuousMassInverse
    mortar_routes: tuple[NodalDGMortarRoute, ...]
    hybrid_boundary_routes: tuple[NodalDGHybridBoundaryRoute, ...]
    three_dimensional_interface_routes: tuple[NodalDGThreeDimensionalInterfaceRoute, ...]
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
        if boundaries.periodic_pairs:
            raise ValueError("Initial nodal DG requires physical exterior patches.")
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
                "triangle",
                "quadrilateral",
                "tetrahedron",
                "prism",
                "pyramid",
            )
            for kind in cell_kinds
        ):
            raise ValueError(
                "Nodal DG supports triangle, quadrilateral, tetrahedron, prism, "
                "or pyramid cells."
            )
        if len(blocks) > 1:
            dimension = discretization.mesh.topological_dimension
            compatible = (
                dimension == 2
                and all(kind in ("triangle", "quadrilateral") for kind in cell_kinds)
            ) or (
                dimension == 3
                and all(
                    kind in ("tetrahedron", "prism", "pyramid") for kind in cell_kinds
                )
            )
            if not compatible:
                raise ValueError("Mixed nodal DG cell kinds are incompatible.")
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
            if not isinstance(system, CompressibleNavierStokesSystem):
                raise TypeError(
                    "Nodal DG viscosity requires CompressibleNavierStokesSystem."
                )
            if (
                len(blocks) != 1
                or blocks[0].cell_kind != "tetrahedron"
                or blocks[0].cell_count != 1
                or int(discretization.interior_facet_domain.entity_indices.shape[0]) != 0
            ):
                raise ValueError(
                    "Initial nodal LDG viscosity requires one physical-boundary "
                    "tetrahedron."
                )
        elif isinstance(system, CompressibleNavierStokesSystem):
            raise ValueError("Compressible Navier–Stokes nodal DG requires viscosity.")
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
        same_block_domain = _same_block_interior_domain(discretization)
        mortar_routes = _mixed_mortar_routes(
            discretization, field_index, interior_degrees
        )
        hybrid_boundary_routes = _hybrid_boundary_routes(
            discretization,
            field_index,
            boundaries,
            max(exterior_degrees.values()),
        )

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
            del gradients, points, test_basis
            state = values[0]
            flux = jnp.stack(
                tuple(
                    system.physical_flux(state, axis, context.user_args)
                    for axis in range(system.dimension)
                ),
                axis=-1,
            )
            return -oe.contract(
                "cq,cqid,cqvd->civ",
                physical_weights,
                test_gradients,
                flux,
                backend="jax",
            )

        def interface_kernel(plus_values, minus_values, points, weights, normal, context):
            del points, weights
            numerical = method.interface_flux.normal_face_flux(
                system,
                plus_values[0],
                minus_values[0],
                normal,
                context.user_args,
            ).normal_flux
            return numerical, -numerical

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
        custom_boundaries = cell_kinds[0] in ("prism", "pyramid") or isinstance(
            discretization.mesh.connectivity, PolyhedralConnectivity
        )
        for patch in () if custom_boundaries else boundaries.patches:

            def boundary_kernel(boundary):
                def kernel(plus_values, points, weights, normal, context):
                    del weights
                    plus = plus_values[0]
                    if isinstance(boundary, PrescribedNormalFluxBoundary):
                        return boundary.normal_flux(
                            context.time,
                            plus,
                            points,
                            normal,
                            context.user_args,
                        )
                    minus = boundary.exterior_state(
                        system,
                        context.time,
                        plus,
                        points,
                        normal,
                        0,
                        context.user_args,
                    )
                    return method.interface_flux.normal_face_flux(
                        system,
                        plus,
                        minus,
                        normal,
                        context.user_args,
                    ).normal_flux

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
                return -oe.contract(
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
        self.hybrid_boundary_routes = hybrid_boundary_routes
        self.three_dimensional_interface_routes = three_dimensional_interface_routes
        self.report = report
        self.dynamics_id = canonical_fingerprint(
            {
                "kind": "prepared-nodal-dg-conservation",
                "system": system.system_id,
                "discretization": discretization.prepared_id,
                "method": method.method_id,
                "boundaries": boundaries.boundary_set_id,
                "mortars": tuple(route.route_id for route in mortar_routes),
                "hybrid_boundaries": tuple(
                    route.route_id for route in hybrid_boundary_routes
                ),
                "three_dimensional_interfaces": tuple(
                    route.route_id for route in three_dimensional_interface_routes
                ),
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
            plus = oe.contract(
                "qi,iv->qv",
                route.owner_basis,
                state[route.owner_dofs],
                backend="jax",
            )
            minus = oe.contract(
                "qi,iv->qv",
                route.neighbour_basis,
                state[route.neighbour_dofs],
                backend="jax",
            )
            flux = self.method.interface_flux.normal_face_flux(
                self.system,
                plus,
                minus,
                route.normal,
                context.user_args,
            ).normal_flux
            owner = oe.contract(
                "q,qi,qv->iv",
                route.physical_weights,
                route.owner_basis,
                flux,
                backend="jax",
            )
            neighbour = -oe.contract(
                "q,qi,qv->iv",
                route.physical_weights,
                route.neighbour_basis,
                flux,
                backend="jax",
            )
            residual = residual.at[route.owner_dofs].add(owner)
            residual = residual.at[route.neighbour_dofs].add(neighbour)
        return residual

    def _viscous_weak_residual(
        self, state: Array, context: FiniteElementExecutionContext, /
    ) -> Array:
        if self.method.viscous is None:
            return jnp.zeros_like(state)
        block = self.discretization.mesh.blocks[0]
        element = self.discretization.elements[0][0]
        routes = self.mass_inverse.routes[0]
        local = state[routes]
        volume_rule, _facet_rule = _rules(
            block.cell_kind,
            self.report.volume_quadrature.selected_degree,
            self.report.exterior_quadrature.selected_degree,
        )
        from ...integration._rules import reference_rule_data

        volume_data = reference_rule_data(volume_rule)
        volume_geometry = self.discretization.evaluate_block_geometry(
            self.discretization.field_spaces[0].name,
            0,
            context.runtime.coordinates,
            volume_data.points,
            volume_data.weights,
        )
        nodal_geometry = self.discretization.evaluate_block_geometry(
            self.discretization.field_spaces[0].name,
            0,
            context.runtime.coordinates,
            element.reference_nodes,
            jnp.ones((element.local_dof_count,), dtype=state.dtype),
        )
        raw_gradient = oe.contract(
            "cqid,civ->cqvd",
            nodal_geometry.physical_gradients,
            local,
            backend="jax",
        )
        correction_dual = jnp.zeros(
            state.shape + (self.system.dimension,), dtype=state.dtype
        )
        for route in self.hybrid_boundary_routes:
            if isinstance(route.boundary, PrescribedNormalFluxBoundary):
                raise ValueError("Nodal LDG viscosity requires a boundary state closure.")
            plus = oe.contract(
                "qi,iv->qv",
                route.basis_values,
                state[route.owner_dofs],
                backend="jax",
            )
            minus = route.boundary.exterior_state(
                self.system,
                context.time,
                plus,
                route.physical_points,
                route.normal,
                0,
                context.user_args,
            )
            common = 0.5 * (plus + minus) + self.method.viscous.beta * (plus - minus)
            local_dual = oe.contract(
                "q,qi,qv,qd->ivd",
                route.physical_weights,
                route.basis_values,
                common - plus,
                route.normal,
                backend="jax",
            )
            correction_dual = correction_dual.at[route.owner_dofs].add(local_dual)
        corrected_global = (
            jnp.zeros(state.shape + (self.system.dimension,), dtype=state.dtype)
            .at[routes]
            .set(raw_gradient)
        )
        corrected_global = corrected_global + self.mass_inverse.apply(correction_dual)
        corrected_local = corrected_global[routes]
        values = oe.contract(
            "qi,civ->cqv",
            volume_geometry.basis_values,
            local,
            backend="jax",
        )
        gradients = oe.contract(
            "qi,civd->cqvd",
            volume_geometry.basis_values,
            corrected_local,
            backend="jax",
        )
        viscous_flux = self.system.viscous_flux(values, gradients, context.user_args)
        local_residual = oe.contract(
            "cq,cqid,cqvd->civ",
            volume_geometry.physical_weights,
            volume_geometry.physical_gradients,
            viscous_flux,
            backend="jax",
        )
        residual = jnp.zeros_like(state).at[routes].set(local_residual)
        for route in self.hybrid_boundary_routes:
            plus = oe.contract(
                "qi,iv->qv",
                route.basis_values,
                state[route.owner_dofs],
                backend="jax",
            )
            gradient = oe.contract(
                "qi,ivd->qvd",
                route.basis_values,
                corrected_global[route.owner_dofs],
                backend="jax",
            )
            minus = route.boundary.exterior_state(
                self.system,
                context.time,
                plus,
                route.physical_points,
                route.normal,
                0,
                context.user_args,
            )
            plus_flux = self.system.viscous_normal_flux(
                plus, gradient, route.normal, context.user_args
            )
            minus_flux = self.system.viscous_normal_flux(
                minus, gradient, route.normal, context.user_args
            )
            common = (
                0.5 * (plus_flux + minus_flux)
                + self.method.viscous.beta * (plus_flux - minus_flux)
                + self.method.viscous.penalty * (minus - plus)
            )
            local_boundary = oe.contract(
                "q,qi,qv->iv",
                route.physical_weights,
                route.basis_values,
                common,
                backend="jax",
            )
            residual = residual.at[route.owner_dofs].add(-local_boundary)
        return residual

    def _mortar_residual(
        self, state: Array, context: FiniteElementExecutionContext, /
    ) -> Array:
        residual = jnp.zeros_like(state)
        for route in self.mortar_routes:
            plus = route.mortar.interpolate_left(state[route.owner_dofs])
            minus = route.mortar.interpolate_right(state[route.neighbour_dofs])
            flux = self.method.interface_flux.normal_face_flux(
                self.system,
                plus,
                minus,
                route.normal,
                context.user_args,
            ).normal_flux
            owner, neighbour = route.mortar.conservative_flux_contributions(flux)
            residual = residual.at[route.owner_dofs].add(owner)
            residual = residual.at[route.neighbour_dofs].add(neighbour)
        return residual

    def _hybrid_boundary_residual(
        self, state: Array, context: FiniteElementExecutionContext, /
    ) -> Array:
        residual = jnp.zeros_like(state)
        if not isinstance(
            self.discretization.mesh.connectivity, PolyhedralConnectivity
        ) and self.discretization.mesh.blocks[0].cell_kind not in ("prism", "pyramid"):
            return residual
        for route in self.hybrid_boundary_routes:
            plus = oe.contract(
                "qi,iv->qv",
                route.basis_values,
                state[route.owner_dofs],
                backend="jax",
            )
            if isinstance(route.boundary, PrescribedNormalFluxBoundary):
                flux = route.boundary.normal_flux(
                    context.time,
                    plus,
                    route.physical_points,
                    route.normal,
                    context.user_args,
                )
            else:
                minus = route.boundary.exterior_state(
                    self.system,
                    context.time,
                    plus,
                    route.physical_points,
                    route.normal,
                    0,
                    context.user_args,
                )
                flux = self.method.interface_flux.normal_face_flux(
                    self.system,
                    plus,
                    minus,
                    route.normal,
                    context.user_args,
                ).normal_flux
            local = oe.contract(
                "q,qi,qv->iv",
                route.physical_weights,
                route.basis_values,
                flux,
                backend="jax",
            )
            residual = residual.at[route.owner_dofs].add(local)
        return residual

    def weak_residual(self, time: Array, state: ArrayLike, args: Any = None, /) -> Array:
        value = self._state(state)
        context = self._context(jnp.asarray(time), args)
        residual = self.compiled_finite_element_problem.residual(value, context)
        return (
            residual
            + self._mortar_residual(value, context)
            + self._three_dimensional_interface_residual(value, context)
            + self._hybrid_boundary_residual(value, context)
            + self._viscous_weak_residual(value, context)
        )

    def mass_inverted_rate(
        self, time: Array, state: ArrayLike, args: Any = None, /
    ) -> Array:
        residual = self.weak_residual(time, state, args)
        return -self.mass_inverse.apply(residual)

    def __call__(self, time: Array, state: ArrayLike, args: Any = None) -> Array:
        return self.mass_inverted_rate(time, state, args)

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
            weighted_value = oe.contract(
                "cij,cjv->civ", matrices, local_value, backend="jax"
            )
            weighted_rate = oe.contract(
                "cij,cjv->civ", matrices, local_rate, backend="jax"
            )
            total_integral = total_integral + jnp.sum(weighted_value, axis=(0, 1))
            conservation_rate = conservation_rate + jnp.sum(weighted_rate, axis=(0, 1))
        admissible = (
            None
            if self.entropy_pair is None
            else jnp.all(self.entropy_pair.admissible(value))
        )
        diagnostics = NodalDGConservationDiagnostics(
            total_integral,
            conservation_rate,
            admissible,
            self.method.method_id,
        )
        return rate, diagnostics

    def linearize(self, time: Array, state: ArrayLike, args: Any = None, /):
        value = self._state(state)
        residual, pushforward = jax.linearize(
            lambda candidate: self(time, candidate, args), value
        )
        _, pullback = jax.vjp(lambda candidate: self(time, candidate, args), value)
        return residual, pushforward, pullback


__all__ = [
    "NodalDGConservationDiagnostics",
    "NodalDGConservationMethodPlan",
    "NodalDGPreparationReport",
    "PreparedNodalDGConservationDynamics",
]
