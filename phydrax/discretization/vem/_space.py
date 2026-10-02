#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import cast

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._precision import PrecisionEvidenceEnvelope
from ..._strict import StrictModule
from ...linalg import ArraySpace
from ...typing import checked, parse
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
from .._polygon_domains import polygon_integration_domains
from .._polygon_geometry import (
    evaluate_polygon_geometry,
    polygon_cubature,
    PolygonAdmissibilityPolicy,
    PolygonCubature,
    PolygonGeometry,
    PolygonTriangulation,
    prepare_polygon_triangulation,
)
from .._polygon_query import (
    polygon_trace_action,
    polygonal_connectivity_of,
    prepare_polygon_facet_sites,
    subset_polygon_domain,
)
from .._side_actions import FacetTraceRule, PreparedTraceAction, SideTraceQuantity
from .._spaces import (
    BlockDofLayout,
    DiscreteFieldSpace,
    EntityDofLayout,
    FieldConformity,
    FieldRepresentation,
)
from .._support import DiscreteSupport
from .._topology import EntitySelection
from .._views import FieldTraceSide
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


def _capabilities(
    field: VirtualElementFieldSpec, /
) -> tuple[DiscretizationCapability, ...]:
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

    @checked
    def __init__(
        self,
        mesh: CellMesh,
        field: VirtualElementFieldSpec,
        /,
        *,
        precision_policy: VirtualElementPrecisionPolicy | None = None,
        admissibility_policy: PolygonAdmissibilityPolicy | None = None,
        resource_budget: VirtualElementResourceBudget | None = None,
    ) -> None:
        if mesh.topological_dimension != 2 or mesh.ambient_dimension != 2:
            raise ValueError("Virtual elements currently require a planar 2-D CellMesh.")
        if not isinstance(mesh.connectivity, PolygonalConnectivity):
            raise TypeError("Virtual elements require polygonal connectivity.")
        if field.element.form_type.dimension != mesh.topological_dimension:
            raise ValueError("Virtual-element form dimension must match the mesh.")
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
    support: DiscreteSupport
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

    @checked
    def __init__(
        self, plan: VirtualElementPlan, /, *, numeric_version: str = "0"
    ) -> None:
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
                    mesh.coordinates.shape[0],
                    dof_map.vertex_dof_count,
                    dofs_per_entity=vertex_width,
                )
            )
            names.append("vertices")
        # VirtualElementPlan rejects meshes without PolygonalConnectivity.
        edge_count = cast(PolygonalConnectivity, mesh.connectivity).edges.shape[0]
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
        representations: dict[str, FieldRepresentation] = {
            "ConformingH1": "functional",
            "ConformingHdiv": "flux_moment",
            "ConformingHcurl": "circulation_moment",
            "DiscontinuousL2": "polynomial_moment",
        }
        conformities: dict[str, FieldConformity] = {
            "ConformingH1": "H1",
            "ConformingHdiv": "Hdiv",
            "ConformingHcurl": "Hcurl",
            "DiscontinuousL2": "L2",
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
                conformity=conformities[element.family],
                form_type=element.form_type,
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
            polygon_integration_domains(mesh)
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

    def edge_trace_routes(self, edges: ArrayLike, /) -> Array:
        """Return the global DOF rows of the edge trace basis on `edges`.

        The result has shape `(edges, k + 1)` and pairs with
        `VirtualElementSpec.edge_trace_basis`: H1 value traces list the
        canonical start vertex, the `k - 1` interior Gauss--Lobatto edge DOFs,
        and the end vertex; H(div)/H(curl) traces list the Legendre moment DOFs
        of each edge by mode. Discontinuous L2 spaces have no trace.
        """
        element = self.field.element
        edges_ = jnp.asarray(edges, dtype=jnp.int32)
        if edges_.ndim != 1:
            raise ValueError("Trace edge indices must be one rank-1 array.")
        offset = self.dof_map.vertex_dof_count
        degree = element.degree
        match element.trace_kind:
            case "value":
                endpoints = jnp.asarray(
                    polygonal_connectivity_of(self.mesh).edges, dtype=jnp.int32
                )[edges_]
                interior = (
                    offset
                    + edges_[:, None] * (degree - 1)
                    + jnp.arange(degree - 1, dtype=jnp.int32)[None, :]
                )
                return jnp.concatenate(
                    (endpoints[:, :1], interior, endpoints[:, 1:]), axis=1
                )
            case "normal" | "tangential":
                modes = jnp.arange(degree + 1, dtype=jnp.int32)
                return offset + edges_[:, None] * (degree + 1) + modes[None, :]
            case "none":
                raise ValueError(
                    "Discontinuous L2 virtual elements have no boundary trace."
                )
            case kind:
                raise ValueError(f"Unknown virtual-element trace kind {kind!r}.")

    def integration_domain(
        self,
        kind: str,
        selection: EntitySelection | None = None,
        /,
    ) -> IntegrationDomain:
        """Cell or facet domain of this space, restricted to a selected entity set.

        ``selection`` masks the entities of the domain's entity set (cells for
        ``"cell"``, edges for the facet kinds); the restricted domain keeps the
        owner/neighbor routes of the selected rows in canonical order.
        """
        match kind:
            case "cell":
                base = self.cell_domain
            case "exterior_facet":
                base = self.exterior_facet_domain
            case "interior_facet":
                base = self.interior_facet_domain
            case _:
                raise ValueError("Unknown virtual-element integration-domain kind.")
        if selection is None:
            return base
        if not isinstance(selection, EntitySelection):
            raise TypeError("selection must be EntitySelection or None.")
        if selection.entity_set_id != base.entity_set_id:
            raise ValueError("Entity selection does not match the domain entity set.")
        mask = np.asarray(selection.mask, dtype=np.bool_)
        entities = np.asarray(base.entity_indices, dtype=np.int32)
        return subset_polygon_domain(base, np.flatnonzero(mask[entities]))

    def prepare_side_trace(
        self,
        field_name: str,
        domain: IntegrationDomain,
        /,
        *,
        rule: FacetTraceRule,
        quantity: SideTraceQuantity = "value",
        side: FieldTraceSide = "owner",
        runtime: VirtualElementRuntimeData | None = None,
    ) -> PreparedTraceAction:
        """Prepare the exact edge trace of the field on selected polygon edges.

        `ConformingH1` fields publish the degree-`k` value trace (Lagrange on
        the Gauss--Lobatto edge nodes); `ConformingHdiv` fields publish the
        normal trace and `ConformingHcurl` fields the tangential trace, both
        from their Legendre moment DOFs and oriented `"outward"` relative to
        the side cell (the tangent is the side cell's counter-clockwise edge
        direction, the `normals` rotated by +90 degrees). These traces are the
        virtual field's own, exact edge traces (`trace_degree=k`), distinct
        from the projected interior channels. Sites follow the owner cell's
        local edge parametrization and are shared by both sides of an interior
        facet. Discontinuous L2 fields have no boundary trace; other
        quantities, `side="neighbor"` on exterior facets, and `side="average"`
        are refused.
        """
        if str(field_name) != self.field.name:
            raise KeyError(f"Unknown virtual-element field {field_name!r}.")
        quantity_ = parse(quantity, SideTraceQuantity, "quantity")
        element = self.field.element
        if element.trace_kind == "none":
            raise ValueError(
                "Discontinuous L2 virtual elements have no L2 boundary trace; their "
                "DOFs are cell moments."
            )
        if quantity_ != element.trace_kind:
            raise ValueError(
                f"{element.family} virtual elements publish "
                f"{element.trace_kind!r} traces, not {quantity_!r} traces."
            )
        runtime_ = self.default_runtime if runtime is None else runtime
        if not virtual_element_runtime_matches(self, runtime_):
            raise ValueError("The VEM trace runtime is incompatible with the space.")
        space = self.field_space.vector_space
        if not isinstance(space, ArraySpace):
            raise TypeError("Virtual-element fields are array valued.")
        sites = prepare_polygon_facet_sites(
            self.mesh, runtime_.coordinates, runtime_.runtime_id, domain, rule, side
        )
        basis = np.asarray(
            element.edge_trace_basis(2.0 * sites.canonical_parameters - 1.0)
        )
        if quantity_ != "value":
            basis = basis * sites.side_signs[:, None, None]
        return polygon_trace_action(
            sites,
            domain,
            rule,
            np.asarray(self.edge_trace_routes(sites.edges)),
            basis,
            space,
            owner_id=self.prepared_id,
            field_space_id=self.field_space.field_space_id,
            quantity=quantity_,
            trace_degree=element.degree,
        )

    @property
    def precision_evidence(self) -> PrecisionEvidenceEnvelope:
        return self.precision_policy.evidence()

    @property
    def resource_evidence_id(self) -> str:
        return self.resource_budget.budget_id

    @property
    def field_space(self) -> DiscreteFieldSpace:
        return self.field_spaces[0]


def virtual_element_runtime_matches(
    discretization: VirtualElementDiscretization,
    runtime: VirtualElementRuntimeData,
    /,
) -> bool:
    """Whether `runtime` realizes the topology, layout, and field of the space."""
    if not isinstance(runtime, VirtualElementRuntimeData):
        raise TypeError("runtime must be VirtualElementRuntimeData.")
    expected_runtime_id = canonical_fingerprint(
        {
            "kind": "virtual-element-runtime",
            "topology": discretization.mesh.topology_id,
            "geometry_layout": discretization.mesh.geometry_layout_id,
            "numeric_version": runtime.numeric_version,
            "field": discretization.field.field_spec_id,
        }
    )
    family = discretization.field.element.family
    return (
        runtime.runtime_id == expected_runtime_id
        and runtime.topology_id == discretization.mesh.topology_id
        and runtime.geometry_layout_id == discretization.mesh.geometry_layout_id
        and all(projection.family == family for projection in runtime.projections)
    )


__all__ = [
    "VirtualElementDiscretization",
    "VirtualElementPlan",
    "VirtualElementRuntimeData",
    "virtual_element_runtime_matches",
]
