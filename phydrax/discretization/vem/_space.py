#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._precision import PrecisionEvidenceEnvelope
from ..._strict import StrictModule
from ...linalg import ArraySpace
from .._cell_complex import PolygonalConnectivity
from .._cell_mesh import CellMesh
from .._core import (
    DiscretizationCapability,
    DiscretizationKey,
    DiscretizationRole,
    PreparationReport,
)
from .._integration_domain import IntegrationDomain
from .._lifecycle import AbstractDiscretizationPlan, AbstractPreparedDiscretization
from .._measure import DiscreteMeasure
from .._polygon_geometry import (
    evaluate_polygon_geometry,
    polygon_cubature,
    PolygonAdmissibilityPolicy,
    PolygonCubature,
    PolygonGeometry,
    PolygonTriangulation,
    prepare_polygon_triangulation,
)
from .._spaces import BlockDofLayout, DiscreteFieldSpace, EntityDofLayout
from ._dofs import VirtualElementDofMap
from ._precision import VirtualElementPrecisionPolicy, VirtualElementResourceBudget
from ._projection import (
    prepare_virtual_element_projections,
    VirtualElementProjectionData,
)
from ._spec import VirtualElementFieldSpec


_BASE_CAPABILITIES = (
    DiscretizationCapability.PROJECTION,
    DiscretizationCapability.RECONSTRUCTION,
    DiscretizationCapability.VARIATIONAL_ASSEMBLY,
    DiscretizationCapability.ENTITY_INCIDENCE,
    DiscretizationCapability.GEOMETRY_REFRESH,
    DiscretizationCapability.DIFFERENTIABLE_GEOMETRY,
    DiscretizationCapability.MATRIX_FREE,
    DiscretizationCapability.SPARSE_ASSEMBLY,
)


def _capabilities(field: VirtualElementFieldSpec, /):
    if field.element.trace_kind == "none":
        return _BASE_CAPABILITIES
    return (
        _BASE_CAPABILITIES[:2]
        + (DiscretizationCapability.TRACE,)
        + _BASE_CAPABILITIES[2:3]
        + (DiscretizationCapability.BOUNDARY_INTEGRAL,)
        + _BASE_CAPABILITIES[3:]
    )


def _projector_storage_bytes(
    mesh: CellMesh,
    field: VirtualElementFieldSpec,
    precision: VirtualElementPrecisionPolicy,
    /,
) -> int:
    element = field.element
    polynomial_count = (element.degree + 1) * (element.degree + 2) // 2
    differential_count = element.degree * (element.degree + 1) // 2
    scalar_count = 0
    for block in mesh.blocks:
        local = element.local_dof_count(block.arity)
        if element.family == "ConformingH1":
            per_cell = 3 * local * polynomial_count
        elif element.family in ("ConformingHdiv", "ConformingHcurl"):
            per_cell = local * (4 * polynomial_count + differential_count)
        else:
            per_cell = 2 * polynomial_count * polynomial_count
        scalar_count += block.cell_count * per_cell
    itemsize = max(
        np.dtype(precision.geometry_dtype).itemsize,
        np.dtype(precision.projection_dtype).itemsize,
    )
    return scalar_count * itemsize


class VirtualElementRuntimeData(StrictModule):
    coordinates: Array
    geometries: tuple[PolygonGeometry, ...]
    cubatures: tuple[PolygonCubature, ...]
    projections: tuple[VirtualElementProjectionData, ...]
    topology_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)
    numeric_version: str = eqx.field(static=True)
    runtime_id: str = eqx.field(static=True)


class VirtualElementPlan(AbstractDiscretizationPlan):
    mesh: CellMesh
    field: VirtualElementFieldSpec
    precision_policy: VirtualElementPrecisionPolicy
    admissibility_policy: PolygonAdmissibilityPolicy
    resource_budget: VirtualElementResourceBudget
    key: DiscretizationKey
    capabilities: tuple[DiscretizationCapability, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh: CellMesh,
        field: VirtualElementFieldSpec,
        /,
        *,
        precision_policy: VirtualElementPrecisionPolicy | None = None,
        admissibility_policy: PolygonAdmissibilityPolicy | None = None,
        resource_budget: VirtualElementResourceBudget | None = None,
    ):
        if not isinstance(mesh, CellMesh):
            raise TypeError("mesh must be CellMesh.")
        if mesh.topological_dimension != 2 or mesh.ambient_dimension != 2:
            raise ValueError("Virtual elements currently require a planar 2-D CellMesh.")
        if not isinstance(mesh.connectivity, PolygonalConnectivity):
            raise TypeError("Virtual elements require polygonal connectivity.")
        if not isinstance(field, VirtualElementFieldSpec):
            raise TypeError("field must be VirtualElementFieldSpec.")
        precision = (
            VirtualElementPrecisionPolicy()
            if precision_policy is None
            else precision_policy
        )
        admissibility = (
            PolygonAdmissibilityPolicy()
            if admissibility_policy is None
            else admissibility_policy
        )
        budget = (
            VirtualElementResourceBudget() if resource_budget is None else resource_budget
        )
        if not isinstance(precision, VirtualElementPrecisionPolicy):
            raise TypeError("precision_policy must be VirtualElementPrecisionPolicy.")
        if not isinstance(admissibility, PolygonAdmissibilityPolicy):
            raise TypeError("admissibility_policy must be PolygonAdmissibilityPolicy.")
        if not isinstance(budget, VirtualElementResourceBudget):
            raise TypeError("resource_budget must be VirtualElementResourceBudget.")
        cell_count = sum(block.cell_count for block in mesh.blocks)
        maximum_local = max(
            field.element.local_dof_count(block.arity) for block in mesh.blocks
        )
        if cell_count > budget.maximum_cells:
            raise ValueError("Virtual-element cell budget exceeded.")
        if maximum_local > budget.maximum_local_dofs:
            raise ValueError("Virtual-element local-DOF budget exceeded.")
        projector_bytes = _projector_storage_bytes(mesh, field, precision)
        if projector_bytes > budget.maximum_projector_bytes:
            raise ValueError("Virtual-element projector storage budget exceeded.")
        self.mesh = mesh
        self.field = field
        self.precision_policy = precision
        self.admissibility_policy = admissibility
        self.resource_budget = budget
        self.key = DiscretizationKey("virtual_element", DiscretizationRole.PHYSICAL)
        self.capabilities = _capabilities(field)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "virtual-element-plan",
                "mesh": mesh.topology_id,
                "field": field.field_spec_id,
                "precision": precision.policy_id,
                "admissibility": admissibility.policy_id,
                "budget": budget.budget_id,
            }
        )

    def prepare(self, /, *, numeric_version: str = "0") -> "VirtualElementDiscretization":
        return VirtualElementDiscretization(self, numeric_version=numeric_version)


class VirtualElementDiscretization(AbstractPreparedDiscretization):
    mesh: CellMesh
    field: VirtualElementFieldSpec
    dof_map: VirtualElementDofMap
    triangulations: tuple[PolygonTriangulation, ...]
    default_runtime: VirtualElementRuntimeData
    cell_domain: IntegrationDomain
    exterior_facet_domain: IntegrationDomain
    interior_facet_domain: IntegrationDomain
    key: DiscretizationKey
    support: object
    field_spaces: tuple[DiscreteFieldSpace, ...]
    measures: tuple[DiscreteMeasure, ...]
    precision_policy: VirtualElementPrecisionPolicy
    admissibility_policy: PolygonAdmissibilityPolicy
    resource_budget: VirtualElementResourceBudget
    capabilities: tuple[DiscretizationCapability, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    numeric_version: str = eqx.field(static=True)
    preparation: PreparationReport

    def __init__(self, plan: VirtualElementPlan, /, *, numeric_version: str = "0"):
        if not isinstance(plan, VirtualElementPlan):
            raise TypeError("plan must be VirtualElementPlan.")
        version = str(numeric_version)
        if not version:
            raise ValueError("numeric_version must be non-empty.")
        mesh = plan.mesh
        dof_map = VirtualElementDofMap(mesh, plan.field.element)
        triangulations = tuple(
            prepare_polygon_triangulation(
                np.asarray(mesh.coordinates),
                np.asarray(block.vertices),
                policy=plan.admissibility_policy,
            )
            for block in mesh.blocks
        )
        self.mesh = mesh
        self.field = plan.field
        self.dof_map = dof_map
        self.triangulations = triangulations
        self.key = plan.key
        self.support = mesh.support
        self.precision_policy = plan.precision_policy
        self.admissibility_policy = plan.admissibility_policy
        self.resource_budget = plan.resource_budget
        self.capabilities = plan.capabilities
        self.plan_id = plan.plan_id
        self.numeric_version = version
        self.default_runtime = self._runtime(mesh.coordinates, version)

        element = plan.field.element
        layouts = []
        names = []
        vertex_width = element.vertex_dofs_per_entity
        if vertex_width:
            layouts.append(
                EntityDofLayout(
                    mesh.topology.entity_sets[0].entity_set_id,
                    int(mesh.coordinates.shape[0]),
                    dof_map.vertex_dof_count,
                    dofs_per_entity=vertex_width,
                )
            )
            names.append("vertices")
        edge_count = int(mesh.connectivity.edges.shape[0])
        edge_width = element.edge_dofs_per_entity
        if edge_width:
            layouts.append(
                EntityDofLayout(
                    mesh.topology.entity_sets[1].entity_set_id,
                    edge_count,
                    dof_map.edge_dof_count,
                    dofs_per_entity=edge_width,
                )
            )
            names.append("edges")
        cell_count = mesh.connectivity.cell_count
        cell_width = element.cell_dofs_per_entity
        if cell_width:
            layouts.append(
                EntityDofLayout(
                    mesh.topology.entity_sets[2].entity_set_id,
                    cell_count,
                    dof_map.cell_dof_count,
                    dofs_per_entity=cell_width,
                )
            )
            names.append("cells")
        layout = BlockDofLayout(tuple(names), tuple(layouts))
        vector_space = ArraySpace((dof_map.global_dof_count,))
        representations = {
            "ConformingH1": "functional",
            "ConformingHdiv": "flux_moment",
            "ConformingHcurl": "circulation_moment",
            "DiscontinuousL2": "polynomial_moment",
        }
        trace_space_id = (
            None
            if element.trace_kind == "none"
            else canonical_fingerprint(
                {
                    "kind": "virtual-element-trace-space",
                    "topology": mesh.topology_id,
                    "trace": element.trace_kind,
                    "degree": element.degree,
                }
            )
        )
        self.field_spaces = (
            DiscreteFieldSpace(
                plan.field.name,
                mesh.support.support_id,
                layout,
                vector_space,
                representation=representations[element.family],
                conformity=element.conformity,
                projection_id=canonical_fingerprint(
                    {
                        "kind": "virtual-element-field-projection",
                        "field": plan.field.field_spec_id,
                    }
                ),
                reconstruction_id=canonical_fingerprint(
                    {
                        "kind": "virtual-element-reconstruction",
                        "field": plan.field.field_spec_id,
                    }
                ),
                trace_space_id=trace_space_id,
            ),
        )
        cell_measures = jnp.concatenate(
            tuple(geometry.areas for geometry in self.default_runtime.geometries)
        )
        self.measures = (
            DiscreteMeasure(
                "virtual_element_cell_measure",
                mesh.support.support_id,
                mesh.topology.entity_sets[2].entity_set_id,
                cell_measures,
            ),
        )
        self.cell_domain, self.exterior_facet_domain, self.interior_facet_domain = (
            _integration_domains(mesh)
        )
        projector_bytes = sum(
            int(
                value.dof_matrix.size
                + value.h1_coefficients.size
                + value.l2_coefficients.size
                + value.differential_coefficients.size
            )
            * np.dtype(value.dof_matrix.dtype).itemsize
            for value in self.default_runtime.projections
        )
        if element.family == "ConformingH1":
            diagnostics = (
                "polygon cells are simple and star-shaped under the declared policy",
                "H1 and enhanced L2 projectors are rank-certified",
                "local-to-global functional DOF routes are fixed",
            )
        elif element.trace_kind != "none":
            diagnostics = (
                "polygon cells are simple and star-shaped under the declared policy",
                f"{element.family} polynomial projectors are rank-certified",
                f"{element.trace_kind} trace topology and local orientations are fixed",
                "local-to-global functional DOF routes are fixed",
            )
        else:
            diagnostics = (
                "polygon cells are simple and star-shaped under the declared policy",
                "cell-local L2 polynomial projectors are rank-certified",
                "local-to-global cell-moment DOF routes are fixed",
            )
        self.preparation = PreparationReport(
            capabilities=plan.capabilities,
            diagnostics=diagnostics,
            resource_counts={
                "cells": cell_count,
                "edges": edge_count,
                "global_dofs": dof_map.global_dof_count,
                "projector_bytes": projector_bytes,
            },
        )
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-virtual-element",
                "plan": plan.plan_id,
                "runtime": self.default_runtime.runtime_id,
                "dof_map": dof_map.dof_map_id,
                "preparation": self.preparation.report_id,
            }
        )

    def _runtime(
        self,
        coordinates: ArrayLike,
        numeric_version: str,
    ) -> VirtualElementRuntimeData:
        points = self.precision_policy.geometry(coordinates)
        if points.shape != self.mesh.coordinates.shape:
            raise ValueError(
                "Virtual-element geometry refresh must preserve coordinates."
            )
        geometries = []
        cubatures = []
        projections = []
        for block, triangulation in zip(
            self.mesh.blocks, self.triangulations, strict=True
        ):
            geometry = evaluate_polygon_geometry(
                points,
                block.vertices,
                triangulation,
                policy=self.admissibility_policy,
                geometry_id=f"virtual-element:{block.block_id}:{numeric_version}",
            )
            cubature = polygon_cubature(
                geometry, triangulation, 2 * self.field.element.degree
            )
            projection = prepare_virtual_element_projections(
                geometry, cubature, self.field.element
            )
            geometries.append(geometry)
            cubatures.append(cubature)
            projections.append(projection)
        layout_id = self.mesh.geometry_layout_id
        return VirtualElementRuntimeData(
            coordinates=points,
            geometries=tuple(geometries),
            cubatures=tuple(cubatures),
            projections=tuple(projections),
            topology_id=self.mesh.topology_id,
            geometry_layout_id=layout_id,
            numeric_version=str(numeric_version),
            runtime_id=canonical_fingerprint(
                {
                    "kind": "virtual-element-runtime",
                    "topology": self.mesh.topology_id,
                    "geometry_layout": layout_id,
                    "numeric_version": str(numeric_version),
                    "field": self.field.field_spec_id,
                }
            ),
        )

    def prepare_runtime(
        self,
        coordinates: ArrayLike,
        /,
        *,
        numeric_version: str,
    ) -> VirtualElementRuntimeData:
        version = str(numeric_version)
        if not version:
            raise ValueError("numeric_version must be non-empty.")
        return self._runtime(coordinates, version)

    @property
    def precision_evidence(self) -> PrecisionEvidenceEnvelope:
        return self.precision_policy.evidence()

    @property
    def resource_evidence_id(self) -> str:
        return self.resource_budget.budget_id

    @property
    def field_space(self) -> DiscreteFieldSpace:
        return self.field_spaces[0]


def _integration_domains(
    mesh: CellMesh, /
) -> tuple[IntegrationDomain, IntegrationDomain, IntegrationDomain]:
    connectivity = mesh.connectivity
    cell_edges = np.asarray(connectivity.cell_edges, dtype=np.int32)
    valid = np.asarray(connectivity.cell_edge_valid, dtype=bool)
    edge_count = int(connectivity.edges.shape[0])
    owner = np.full((edge_count,), -1, dtype=np.int32)
    neighbour = np.full((edge_count,), -1, dtype=np.int32)
    owner_local = np.full((edge_count,), -1, dtype=np.int32)
    neighbour_local = np.full((edge_count,), -1, dtype=np.int32)
    for cell in range(cell_edges.shape[0]):
        for local in range(cell_edges.shape[1]):
            if not valid[cell, local]:
                continue
            edge = int(cell_edges[cell, local])
            if owner[edge] < 0:
                owner[edge] = cell
                owner_local[edge] = local
            else:
                neighbour[edge] = cell
                neighbour_local[edge] = local
    exterior = np.flatnonzero(neighbour < 0).astype(np.int32)
    interior = np.flatnonzero(neighbour >= 0).astype(np.int32)
    cell_domain = IntegrationDomain(
        "cell",
        np.arange(connectivity.cell_count, dtype=np.int32),
        mesh.support.support_id,
        mesh.topology.entity_sets[2].entity_set_id,
    )
    exterior_domain = IntegrationDomain(
        "exterior_facet",
        exterior,
        mesh.support.support_id,
        mesh.topology.entity_sets[1].entity_set_id,
        owner_cells=owner[exterior],
        owner_local_entities=owner_local[exterior],
    )
    interior_domain = IntegrationDomain(
        "interior_facet",
        interior,
        mesh.support.support_id,
        mesh.topology.entity_sets[1].entity_set_id,
        owner_cells=owner[interior],
        neighbour_cells=neighbour[interior],
        owner_local_entities=owner_local[interior],
        neighbour_local_entities=neighbour_local[interior],
    )
    return cell_domain, exterior_domain, interior_domain


__all__ = [
    "VirtualElementDiscretization",
    "VirtualElementPlan",
    "VirtualElementRuntimeData",
]
