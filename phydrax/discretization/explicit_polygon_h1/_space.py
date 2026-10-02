#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._precision import PrecisionEvidenceEnvelope
from ..._strict import StrictModule
from ...linalg import ArraySpace, BlockSpace
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
from .._lifecycle import AbstractDiscretizationPlan, validate_prepared_metadata
from .._local_variational import (
    AbstractPreparedLocalDiscretization,
    LocalFieldBinding,
    LocalVariationalCapabilities,
    PreparedLocalRegion,
)
from .._measure import DiscreteMeasure
from .._polygon_domains import polygon_integration_domains
from .._polygon_geometry import (
    evaluate_polygon_geometry,
    PolygonAdmissibilityPolicy,
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
from .._spaces import DiscreteFieldSpace, EntityDofLayout
from .._support import DiscreteSupport
from .._topology import EntitySelection
from .._views import FieldTraceSide
from ._basis import ExplicitPolygonH1BlockData, prepare_explicit_polygon_h1_basis
from ._dofs import ExplicitPolygonH1DofMap
from ._precision import ExplicitPolygonH1PrecisionPolicy
from ._spec import (
    ExplicitPolygonH1FieldSpec,
    ExplicitPolygonH1QuadraturePolicy,
    ExplicitPolygonH1QualificationPolicy,
    ExplicitPolygonH1ResourceBudget,
)


_CAPABILITIES = (
    DiscretizationCapability.RECONSTRUCTION,
    DiscretizationCapability.TRACE,
    DiscretizationCapability.BOUNDARY_INTEGRAL,
    DiscretizationCapability.VARIATIONAL_ASSEMBLY,
    DiscretizationCapability.ENTITY_INCIDENCE,
    DiscretizationCapability.GEOMETRY_REFRESH,
    DiscretizationCapability.DIFFERENTIABLE_GEOMETRY,
    DiscretizationCapability.MATRIX_FREE,
)


def _estimated_storage_bytes(
    mesh: CellMesh,
    local_width: int,
    quadrature: ExplicitPolygonH1QuadraturePolicy,
    precision: ExplicitPolygonH1PrecisionPolicy,
    /,
) -> tuple[int, int]:
    retained_scalars = 0
    workspace_scalars = 0
    for block in mesh.blocks:
        cells = block.cell_count
        arity = block.arity
        points = arity * quadrature.cell_order**2
        retained_scalars += cells * (
            (arity + 1) * arity + points * local_width * 3 + points * (2 + 1 + 4 + 4)
        )
        workspace_scalars += cells * (arity + 1) ** 2 * 4
    itemsize = max(
        np.dtype(precision.geometry_dtype).itemsize,
        np.dtype(precision.basis_dtype).itemsize,
        np.dtype(precision.factorization_dtype).itemsize,
    )
    return retained_scalars * itemsize, workspace_scalars * itemsize


class ExplicitPolygonH1RuntimeData(StrictModule):
    """Geometry-dependent explicit bases over one fixed polygon topology."""

    coordinates: Array
    geometries: tuple[PolygonGeometry, ...]
    bases: tuple[ExplicitPolygonH1BlockData, ...]
    plan_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)
    numeric_version: str = eqx.field(static=True)
    runtime_id: str = eqx.field(static=True)


class ExplicitPolygonH1Plan(AbstractDiscretizationPlan):
    """Deterministic lowest-order explicit H1 basis on star-shaped polygons."""

    mesh: CellMesh
    field: ExplicitPolygonH1FieldSpec
    quadrature_policy: ExplicitPolygonH1QuadraturePolicy
    precision_policy: ExplicitPolygonH1PrecisionPolicy
    qualification_policy: ExplicitPolygonH1QualificationPolicy
    admissibility_policy: PolygonAdmissibilityPolicy
    resource_budget: ExplicitPolygonH1ResourceBudget
    key: DiscretizationKey
    capabilities: tuple[DiscretizationCapability, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        mesh: CellMesh,
        field: ExplicitPolygonH1FieldSpec,
        /,
        *,
        quadrature_policy: ExplicitPolygonH1QuadraturePolicy | None = None,
        precision_policy: ExplicitPolygonH1PrecisionPolicy | None = None,
        qualification_policy: ExplicitPolygonH1QualificationPolicy | None = None,
        admissibility_policy: PolygonAdmissibilityPolicy | None = None,
        resource_budget: ExplicitPolygonH1ResourceBudget | None = None,
    ) -> None:
        if mesh.topological_dimension != 2 or mesh.ambient_dimension != 2:
            raise ValueError("Explicit polygon H1 requires a planar 2-D CellMesh.")
        if not isinstance(mesh.connectivity, PolygonalConnectivity):
            raise TypeError("Explicit polygon H1 requires polygon connectivity.")
        quadrature = (
            ExplicitPolygonH1QuadraturePolicy()
            if quadrature_policy is None
            else quadrature_policy
        )
        precision = (
            ExplicitPolygonH1PrecisionPolicy()
            if precision_policy is None
            else precision_policy
        )
        qualification = (
            ExplicitPolygonH1QualificationPolicy()
            if qualification_policy is None
            else qualification_policy
        )
        admissibility = (
            PolygonAdmissibilityPolicy()
            if admissibility_policy is None
            else admissibility_policy
        )
        budget = (
            ExplicitPolygonH1ResourceBudget()
            if resource_budget is None
            else resource_budget
        )
        if not isinstance(quadrature, ExplicitPolygonH1QuadraturePolicy):
            raise TypeError("quadrature_policy has the wrong type.")
        if not isinstance(precision, ExplicitPolygonH1PrecisionPolicy):
            raise TypeError("precision_policy has the wrong type.")
        if not isinstance(qualification, ExplicitPolygonH1QualificationPolicy):
            raise TypeError("qualification_policy has the wrong type.")
        if not isinstance(admissibility, PolygonAdmissibilityPolicy):
            raise TypeError("admissibility_policy has the wrong type.")
        if not isinstance(budget, ExplicitPolygonH1ResourceBudget):
            raise TypeError("resource_budget has the wrong type.")
        cells = sum(block.cell_count for block in mesh.blocks)
        maximum_arity = max(block.arity for block in mesh.blocks)
        if cells > budget.maximum_cells:
            raise ValueError("Explicit polygon cell budget exceeded.")
        if maximum_arity > budget.maximum_arity:
            raise ValueError("Explicit polygon arity budget exceeded.")
        retained, workspace = _estimated_storage_bytes(
            mesh, maximum_arity, quadrature, precision
        )
        if retained > budget.maximum_retained_bytes:
            raise ValueError("Explicit polygon retained-storage budget exceeded.")
        if workspace > budget.maximum_workspace_bytes:
            raise ValueError("Explicit polygon preparation workspace budget exceeded.")
        self.mesh = mesh
        self.field = field
        self.quadrature_policy = quadrature
        self.precision_policy = precision
        self.qualification_policy = qualification
        self.admissibility_policy = admissibility
        self.resource_budget = budget
        self.key = DiscretizationKey("explicit_polygon_h1", DiscretizationRole.PHYSICAL)
        self.capabilities = _CAPABILITIES
        self.plan_id = canonical_fingerprint(
            {
                "kind": "explicit-polygon-h1-plan",
                "mesh": mesh.mesh_id,
                "field": field.field_spec_id,
                "quadrature": quadrature.policy_id,
                "precision": precision.policy_id,
                "qualification": qualification.policy_id,
                "admissibility": admissibility.policy_id,
                "budget": budget.budget_id,
            }
        )

    def prepare(
        self, /, *, numeric_version: str = "0"
    ) -> ExplicitPolygonH1Discretization:
        return ExplicitPolygonH1Discretization(self, numeric_version=numeric_version)


class ExplicitPolygonH1Discretization(AbstractPreparedLocalDiscretization):
    """Prepared local discretization backed by condensed polygon fan bases."""

    mesh: CellMesh
    field: ExplicitPolygonH1FieldSpec
    dof_map: ExplicitPolygonH1DofMap
    triangulations: tuple[PolygonTriangulation, ...]
    default_runtime: ExplicitPolygonH1RuntimeData
    cell_domain: IntegrationDomain
    exterior_facet_domain: IntegrationDomain
    interior_facet_domain: IntegrationDomain
    key: DiscretizationKey
    support: DiscreteSupport
    field_spaces: tuple[DiscreteFieldSpace, ...]
    block_space: BlockSpace
    measures: tuple[DiscreteMeasure, ...]
    binding: LocalFieldBinding
    quadrature_policy: ExplicitPolygonH1QuadraturePolicy
    precision_policy: ExplicitPolygonH1PrecisionPolicy
    qualification_policy: ExplicitPolygonH1QualificationPolicy
    admissibility_policy: PolygonAdmissibilityPolicy
    resource_budget: ExplicitPolygonH1ResourceBudget
    capabilities: tuple[DiscretizationCapability, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    numeric_version: str = eqx.field(static=True)
    preparation: PreparationReport

    @checked
    def __init__(
        self, plan: ExplicitPolygonH1Plan, /, *, numeric_version: str = "0"
    ) -> None:
        version = str(numeric_version)
        if not version:
            raise ValueError("numeric_version must be non-empty.")
        mesh = plan.mesh
        dof_map = ExplicitPolygonH1DofMap(mesh)
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
        self.quadrature_policy = plan.quadrature_policy
        self.precision_policy = plan.precision_policy
        self.qualification_policy = plan.qualification_policy
        self.admissibility_policy = plan.admissibility_policy
        self.resource_budget = plan.resource_budget
        self.key = plan.key
        self.support = mesh.support
        self.capabilities = plan.capabilities
        self.plan_id = plan.plan_id
        self.numeric_version = version
        self.default_runtime = self._runtime(mesh.coordinates, version)
        layout = EntityDofLayout(
            mesh.topology.entity_sets[0].entity_set_id,
            mesh.coordinates.shape[0],
            dof_map.global_dof_count,
            component_shape=plan.field.component_shape,
        )
        vector_space = ArraySpace(
            (dof_map.global_dof_count,) + plan.field.component_shape,
            dtype=plan.precision_policy.output_dtype,
        )
        field_space = DiscreteFieldSpace(
            plan.field.name,
            mesh.support.support_id,
            layout,
            vector_space,
            representation="point_value",
            conformity="H1",
            reconstruction_id=canonical_fingerprint(
                {
                    "kind": "explicit-polygon-h1-reconstruction",
                    "field": plan.field.field_spec_id,
                }
            ),
            trace_space_id=canonical_fingerprint(
                {"kind": "explicit-polygon-h1-trace", "field": plan.field.field_spec_id}
            ),
        )
        self.field_spaces = (field_space,)
        self.block_space = BlockSpace((vector_space,), names=(plan.field.name,))
        self.binding = LocalFieldBinding(
            plan.field.name,
            field_space,
            component_shape=plan.field.component_shape,
            public_shape=plan.field.component_shape,
            execution_shape=plan.field.component_shape,
            local_width=dof_map.local_width,
            layout_id=canonical_fingerprint(
                {
                    "kind": "explicit-polygon-h1-local-layout",
                    "dof_map": dof_map.dof_map_id,
                }
            ),
        )
        self.cell_domain, self.exterior_facet_domain, self.interior_facet_domain = (
            polygon_integration_domains(mesh)
        )
        cell_measures = jnp.concatenate(
            tuple(geometry.areas for geometry in self.default_runtime.geometries)
        )
        self.measures = (
            DiscreteMeasure(
                "explicit_polygon_h1_cell_measure",
                mesh.support.support_id,
                mesh.topology.entity_sets[2].entity_set_id,
                cell_measures,
            ),
        )
        retained, workspace = _estimated_storage_bytes(
            mesh,
            dof_map.local_width,
            plan.quadrature_policy,
            plan.precision_policy,
        )
        self.preparation = PreparationReport(
            capabilities=plan.capabilities,
            diagnostics=(
                "polygon interfaces have matching edge segmentation",
                "transported witness fans are positive and star-visible",
                "discrete-harmonic condensation is rank-certified",
                "partition, affine reproduction, and exact trace are certified",
            ),
            resource_counts={
                "cells": mesh.connectivity.cell_count,
                "global_dofs": dof_map.global_dof_count,
                "maximum_arity": dof_map.local_width,
                "retained_bytes": retained,
                "workspace_bytes": workspace,
            },
        )
        spaces, measures, capabilities = validate_prepared_metadata(
            key=self.key,
            support=self.support,
            field_spaces=self.field_spaces,
            measures=self.measures,
            capabilities=self.capabilities,
            preparation=self.preparation,
        )
        self.field_spaces = spaces
        self.measures = measures
        self.capabilities = capabilities
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-explicit-polygon-h1",
                "plan": plan.plan_id,
                "runtime": self.default_runtime.runtime_id,
                "dof_map": dof_map.dof_map_id,
                "preparation": self.preparation.report_id,
            }
        )

    def _runtime(
        self, coordinates: ArrayLike, numeric_version: str, /
    ) -> ExplicitPolygonH1RuntimeData:
        points = self.precision_policy.geometry(coordinates)
        if points.shape != self.mesh.coordinates.shape:
            raise ValueError(
                "Explicit polygon geometry refresh must preserve coordinate shape."
            )
        geometries = []
        bases = []
        for block, triangulation in zip(
            self.mesh.blocks, self.triangulations, strict=True
        ):
            geometry = evaluate_polygon_geometry(
                points,
                block.vertices,
                triangulation,
                policy=self.admissibility_policy,
                geometry_id=f"explicit-polygon-h1:{block.block_id}:{numeric_version}",
            )
            basis = prepare_explicit_polygon_h1_basis(
                geometry,
                triangulation,
                self.dof_map.local_width,
                self.quadrature_policy,
                self.precision_policy,
                self.qualification_policy,
            )
            geometries.append(geometry)
            bases.append(basis)
        passed = jnp.all(jnp.concatenate(tuple(basis.evidence.passed for basis in bases)))
        points = eqx.error_if(
            points,
            ~passed,
            "Explicit polygon H1 basis certification failed.",
        )
        return ExplicitPolygonH1RuntimeData(
            coordinates=points,
            geometries=tuple(geometries),
            bases=tuple(bases),
            plan_id=self.plan_id,
            topology_id=self.mesh.topology_id,
            geometry_layout_id=self.mesh.geometry_layout_id,
            numeric_version=str(numeric_version),
            runtime_id=canonical_fingerprint(
                {
                    "kind": "explicit-polygon-h1-runtime",
                    "plan": self.plan_id,
                    "topology": self.mesh.topology_id,
                    "geometry_layout": self.mesh.geometry_layout_id,
                    "numeric_version": str(numeric_version),
                }
            ),
        )

    def prepare_runtime(
        self, coordinates: ArrayLike, /, *, numeric_version: str
    ) -> ExplicitPolygonH1RuntimeData:
        version = str(numeric_version)
        if not version:
            raise ValueError("numeric_version must be non-empty.")
        return self._runtime(coordinates, version)

    def _field_index(self, name: str, /) -> int:
        if str(name) != self.field.name:
            raise KeyError(f"Unknown explicit polygon field {name!r}.")
        return 0

    def local_field_binding(self, name: str, /) -> LocalFieldBinding:
        self._field_index(name)
        return self.binding

    def local_variational_capabilities(self, /) -> LocalVariationalCapabilities:
        from ._actions import ExplicitPolygonH1LocalProvider

        return ExplicitPolygonH1LocalProvider(self).local_variational_capabilities()

    def integration_domain(
        self, kind: str, selection: EntitySelection | None = None, /
    ) -> IntegrationDomain:
        kind_ = str(kind)
        if kind_ == "cell":
            base = self.cell_domain
        elif kind_ == "exterior_facet":
            base = self.exterior_facet_domain
        elif kind_ == "interior_facet":
            base = self.interior_facet_domain
        else:
            raise ValueError("Unknown explicit polygon integration-domain kind.")
        if selection is None:
            return base
        if selection.entity_set_id != base.entity_set_id:
            raise ValueError("Selection does not match the requested polygon domain.")
        rows = np.flatnonzero(
            np.asarray(selection.mask, dtype=np.bool_)[np.asarray(base.entity_indices)]
        )
        return subset_polygon_domain(base, rows)

    def prepare_local_regions(
        self,
        domain: IntegrationDomain,
        /,
        *,
        field_names: tuple[str, ...],
        maximum_derivative_order: int,
        kernel_mode: str,
    ) -> tuple[PreparedLocalRegion, ...]:
        from ._actions import ExplicitPolygonH1LocalProvider

        return ExplicitPolygonH1LocalProvider(self).prepare_local_regions(
            domain,
            field_names=field_names,
            maximum_derivative_order=maximum_derivative_order,
            kernel_mode=kernel_mode,
        )

    def prepare_side_trace(
        self,
        field_name: str,
        domain: IntegrationDomain,
        /,
        *,
        rule: FacetTraceRule,
        quantity: SideTraceQuantity = "value",
        side: FieldTraceSide = "owner",
        runtime: ExplicitPolygonH1RuntimeData | None = None,
    ) -> PreparedTraceAction:
        """Prepare the exact value trace of the field on selected polygon edges.

        The trace on each straight edge is linear in the two edge vertex values
        (`trace_degree=1`), so it is the field's own trace, identical from both
        incident cells. Sites follow the owner cell's local edge
        parametrization at the reference points of `rule` and are the same
        physical points for `side="owner"` and `side="neighbor"`; `normals` are
        unit outward normals of the side cell. Only `quantity="value"` exists
        for this scalar-basis H1 field; `side="neighbor"` requires interior
        facets and `side="average"` is refused (compose both sides instead).
        """
        self._field_index(field_name)
        quantity_ = parse(quantity, SideTraceQuantity, "quantity")
        if quantity_ != "value":
            raise ValueError(
                "Explicit polygon H1 fields publish exact value traces only; "
                f"{quantity_!r} traces belong to vector fields or to compiled "
                "physics owners."
            )
        runtime_ = self.default_runtime if runtime is None else runtime
        self.validate_local_runtime(runtime_)
        space = self.field_space.vector_space
        if not isinstance(space, ArraySpace):
            raise TypeError("Explicit polygon H1 fields are array valued.")
        sites = prepare_polygon_facet_sites(
            self.mesh, runtime_.coordinates, runtime_.runtime_id, domain, rule, side
        )
        edges = np.asarray(polygonal_connectivity_of(self.mesh).edges, dtype=np.int32)
        parameters = sites.canonical_parameters
        return polygon_trace_action(
            sites,
            domain,
            rule,
            edges[sites.edges],
            np.stack((1.0 - parameters, parameters), axis=-1),
            space,
            owner_id=self.prepared_id,
            field_space_id=self.field_space.field_space_id,
            quantity=quantity_,
            trace_degree=1,
        )

    def validate_local_runtime(self, runtime: object, /) -> None:
        if not isinstance(runtime, ExplicitPolygonH1RuntimeData):
            raise TypeError("Explicit polygon execution requires its runtime data.")
        if (
            runtime.plan_id != self.plan_id
            or runtime.topology_id != self.mesh.topology_id
            or runtime.geometry_layout_id != self.mesh.geometry_layout_id
            or runtime.coordinates.shape != self.default_runtime.coordinates.shape
        ):
            raise ValueError("Explicit polygon runtime does not match prepared layout.")

    @property
    def precision_evidence(self) -> PrecisionEvidenceEnvelope:
        return self.precision_policy.evidence()

    @property
    def resource_evidence_id(self) -> str:
        return self.resource_budget.budget_id

    @property
    def field_space(self) -> DiscreteFieldSpace:
        return self.field_spaces[0]


__all__ = [
    "ExplicitPolygonH1Discretization",
    "ExplicitPolygonH1Plan",
    "ExplicitPolygonH1RuntimeData",
]
