#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import ArraySpace, OperatorProperties
from ...sparse import EdgeRelation, RowRelation, SparseLinearMap
from .._cell_complex import PolygonalConnectivity
from .._cell_mesh import CellBlock, CellMesh
from .._core import (
    DiscretizationCapability,
    DiscretizationKey,
    DiscretizationRole,
    PreparationReport,
)
from .._lifecycle import (
    AbstractDiscretizationPlan,
    AbstractPreparedDiscretization,
    validate_prepared_metadata,
)
from .._measure import DiscreteMeasure
from .._spaces import DiscreteFieldSpace, EntityDofLayout
from .._support import DiscreteSupport
from ._precision import FiniteElementPrecisionPolicy
from ._reference import FiniteElementSpec, lagrange_element


class FiniteElementFieldSpec(StrictModule, NonTrainableState):
    """One named field and its reference element on every mesh block."""

    name: str = eqx.field(static=True)
    block_names: tuple[str, ...] = eqx.field(static=True)
    elements: tuple[FiniteElementSpec, ...]
    component_shape: tuple[int, ...] = eqx.field(static=True)
    field_spec_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        elements: FiniteElementSpec | Mapping[str, FiniteElementSpec],
        /,
        *,
        block_names: Sequence[str] | None = None,
        component_shape: Sequence[int] = (),
    ):
        field_name = str(name)
        if not field_name:
            raise ValueError("Finite-element field name must be non-empty.")
        components = tuple(int(size) for size in component_shape)
        if any(size <= 0 for size in components):
            raise ValueError("Field component dimensions must be positive.")
        if isinstance(elements, FiniteElementSpec):
            if block_names is None:
                names = ()
            else:
                names = tuple(str(block) for block in block_names)
            element_values = (elements,) if not names else (elements,) * len(names)
        else:
            items = tuple(
                sorted((str(block), element) for block, element in elements.items())
            )
            if not items:
                raise ValueError("Field element mapping must be non-empty.")
            names = tuple(block for block, _ in items)
            element_values = tuple(element for _, element in items)
        if any(not name_ for name_ in names) or len(set(names)) != len(names):
            raise ValueError("Field block names must be unique and non-empty.")
        if not all(isinstance(element, FiniteElementSpec) for element in element_values):
            raise TypeError("Field elements must be FiniteElementSpec instances.")
        self.name = field_name
        self.block_names = names
        self.elements = element_values
        self.component_shape = components
        self.field_spec_id = canonical_fingerprint(
            {
                "kind": "finite-element-field-spec",
                "name": field_name,
                "blocks": list(names),
                "elements": [element.element_id for element in element_values],
                "component_shape": list(components),
            }
        )

    def resolve(self, mesh: CellMesh, /) -> tuple[FiniteElementSpec, ...]:
        if not self.block_names:
            if len(self.elements) != 1:
                raise ValueError(
                    "Implicit field element assignment requires one element."
                )
            element = self.elements[0]
            resolved = (element,) * len(mesh.blocks)
        else:
            assignments = dict(zip(self.block_names, self.elements, strict=True))
            if set(assignments) != {block.name for block in mesh.blocks}:
                raise ValueError(
                    "Field element assignments must match the mesh block names exactly."
                )
            resolved = tuple(assignments[block.name] for block in mesh.blocks)
        for block, element in zip(mesh.blocks, resolved, strict=True):
            if block.cell_kind != element.cell_kind:
                raise ValueError(
                    f"Element {element.cell_kind!r} is incompatible with block "
                    f"{block.name!r} ({block.cell_kind!r})."
                )
        if any(element.degree == 2 for element in resolved) and not all(
            element.cell_kind == "triangle" and element.degree == 2
            for element in resolved
        ):
            raise ValueError("P2 fields currently require a purely triangular P2 mesh.")
        return resolved


class FiniteElementDofMap(StrictModule, NonTrainableState):
    """Per-block FE local gathers into one global field coordinate array."""

    block_names: tuple[str, ...] = eqx.field(static=True)
    cell_dofs: tuple[Array, ...]
    relations: tuple[RowRelation, ...]
    global_dof_count: int = eqx.field(static=True)
    boundary_dof_mask: Array
    dof_coordinates: Array
    dof_map_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh: CellMesh,
        elements: Sequence[FiniteElementSpec],
        /,
    ):
        resolved = tuple(elements)
        if len(resolved) != len(mesh.blocks):
            raise ValueError("One finite element is required per mesh block.")
        vertex_count = int(mesh.coordinates.shape[0])
        p2 = any(element.degree == 2 for element in resolved)
        if p2:
            if not isinstance(mesh.connectivity, PolygonalConnectivity):
                raise ValueError("P2 DOF maps currently require polygonal connectivity.")
            global_count = vertex_count + int(mesh.connectivity.edges.shape[0])
        else:
            global_count = vertex_count
        block_dofs = []
        relations = []
        cell_offset = 0
        for block, element in zip(mesh.blocks, resolved, strict=True):
            vertices = np.asarray(block.vertices, dtype=np.int32)
            if element.degree == 1:
                local = vertices
            elif block.cell_kind == "triangle" and element.degree == 2:
                connectivity = mesh.connectivity
                if not isinstance(connectivity, PolygonalConnectivity):
                    raise TypeError("Triangle P2 requires polygonal connectivity.")
                edges = np.asarray(connectivity.cell_edges, dtype=np.int32)[
                    cell_offset : cell_offset + block.cell_count, :3
                ]
                local = np.concatenate((vertices, vertex_count + edges), axis=1)
            else:
                raise ValueError("Unsupported finite-element DOF map.")
            block_dofs.append(jnp.asarray(local))
            relations.append(RowRelation(local, source_size=global_count))
            cell_offset += block.cell_count
        boundary = np.zeros((global_count,), dtype=bool)
        boundary[:vertex_count] = np.asarray(
            mesh.topology.entity_sets[0].subset("boundary").mask,
            dtype=bool,
        )
        if p2:
            connectivity = mesh.connectivity
            if not isinstance(connectivity, PolygonalConnectivity):
                raise TypeError("P2 boundary map requires polygonal connectivity.")
            boundary[vertex_count:] = np.asarray(connectivity.boundary_edges, dtype=bool)
        dof_coordinates = np.asarray(mesh.coordinates)
        if p2:
            connectivity = mesh.connectivity
            if not isinstance(connectivity, PolygonalConnectivity):
                raise TypeError("P2 coordinate map requires polygonal connectivity.")
            edge_vertices = np.asarray(connectivity.edges, dtype=np.int32)
            edge_coordinates = np.mean(dof_coordinates[edge_vertices], axis=1)
            dof_coordinates = np.concatenate((dof_coordinates, edge_coordinates), axis=0)
        self.block_names = tuple(block.name for block in mesh.blocks)
        self.cell_dofs = tuple(block_dofs)
        self.relations = tuple(relations)
        self.global_dof_count = global_count
        self.boundary_dof_mask = jnp.asarray(boundary)
        self.dof_coordinates = jnp.asarray(dof_coordinates)
        self.dof_map_id = canonical_fingerprint(
            {
                "kind": "finite-element-dof-map",
                "mesh": mesh.topology_id,
                "elements": [element.element_id for element in resolved],
                "global_dof_count": global_count,
                "cell_dofs": [np.asarray(value).tolist() for value in block_dofs],
            }
        )


class FiniteElementBlockGeometry(StrictModule, NonTrainableState):
    block_name: str = eqx.field(static=True)
    reference_points: Array
    reference_weights: Array
    basis_values: Array
    reference_gradients: Array
    physical_points: Array
    physical_gradients: Array
    physical_weights: Array
    measure: Array


class IntegrationDomain(StrictModule, NonTrainableState):
    """Rule-free resolved selection of cells or codimension-one facets."""

    kind: str = eqx.field(static=True)
    entity_indices: Array
    support_id: str = eqx.field(static=True)
    domain_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: str,
        entity_indices: ArrayLike,
        support_id: str,
        /,
    ):
        kind_ = str(kind)
        if kind_ not in ("cell", "exterior_facet", "interior_facet"):
            raise ValueError("Unsupported finite-element integration-domain kind.")
        indices = np.asarray(entity_indices, dtype=np.int32)
        if (
            indices.ndim != 1
            or np.any(indices < 0)
            or np.unique(indices).size != indices.size
        ):
            raise ValueError(
                "Integration-domain indices must be unique non-negative IDs."
            )
        support = str(support_id)
        if not support:
            raise ValueError("Integration-domain support_id must be non-empty.")
        self.kind = kind_
        self.entity_indices = jnp.asarray(indices)
        self.support_id = support
        self.domain_id = canonical_fingerprint(
            {
                "kind": "finite-element-integration-domain",
                "entity_kind": kind_,
                "indices": indices.tolist(),
                "support": support,
            }
        )


def _validate_mesh_geometry(mesh: CellMesh, /) -> None:
    coordinates = np.asarray(mesh.coordinates, dtype=float)
    for block in mesh.blocks:
        cells = np.asarray(block.vertices, dtype=np.int32)
        points = coordinates[cells]
        if block.cell_kind == "triangle":
            first = points[:, 1] - points[:, 0]
            second = points[:, 2] - points[:, 0]
            determinant = (
                np.sum(first * first, axis=-1) * np.sum(second * second, axis=-1)
                - np.sum(first * second, axis=-1) ** 2
            )
        elif block.cell_kind == "quadrilateral":
            first = points[:, 1] - points[:, 0]
            second = points[:, 3] - points[:, 0]
            determinant = (
                np.sum(first * first, axis=-1) * np.sum(second * second, axis=-1)
                - np.sum(first * second, axis=-1) ** 2
            )
        elif block.cell_kind == "tetrahedron":
            edge_matrix = np.stack(
                (
                    points[:, 1] - points[:, 0],
                    points[:, 2] - points[:, 0],
                    points[:, 3] - points[:, 0],
                ),
                axis=-1,
            )
            gram = np.swapaxes(edge_matrix, -1, -2) @ edge_matrix
            determinant = np.linalg.det(gram)
        else:
            raise ValueError("Unsupported finite-element cell kind.")
        if np.any(~np.isfinite(determinant)) or np.any(determinant <= 0.0):
            raise ValueError(
                "Finite-element cells require positive finite metric determinant."
            )


class FiniteElementPlan(AbstractDiscretizationPlan):
    mesh: CellMesh
    fields: tuple[FiniteElementFieldSpec, ...]
    precision_policy: FiniteElementPrecisionPolicy
    key: DiscretizationKey
    capabilities: tuple[DiscretizationCapability, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh: CellMesh,
        fields: FiniteElementFieldSpec | Sequence[FiniteElementFieldSpec],
        /,
        *,
        precision_policy: FiniteElementPrecisionPolicy | None = None,
    ):
        if not isinstance(mesh, CellMesh):
            raise TypeError("mesh must be a CellMesh.")
        _validate_mesh_geometry(mesh)
        used_vertices = np.unique(
            np.concatenate(
                tuple(
                    np.asarray(block.vertices, dtype=np.int32).reshape((-1,))
                    for block in mesh.blocks
                )
            )
        )
        if used_vertices.size != mesh.coordinates.shape[0]:
            raise ValueError("FiniteElementPlan requires every mesh vertex to be used.")
        field_specs = (
            (fields,) if isinstance(fields, FiniteElementFieldSpec) else tuple(fields)
        )
        if not field_specs or not all(
            isinstance(field, FiniteElementFieldSpec) for field in field_specs
        ):
            raise TypeError("fields must contain FiniteElementFieldSpec instances.")
        names = tuple(field.name for field in field_specs)
        if len(set(names)) != len(names):
            raise ValueError("Finite-element field names must be unique.")
        for field in field_specs:
            field.resolve(mesh)
        precision = (
            FiniteElementPrecisionPolicy()
            if precision_policy is None
            else precision_policy
        )
        if not isinstance(precision, FiniteElementPrecisionPolicy):
            raise TypeError(
                "precision_policy must be FiniteElementPrecisionPolicy or None."
            )
        self.mesh = mesh
        self.fields = field_specs
        self.precision_policy = precision
        self.key = DiscretizationKey("finite_element", DiscretizationRole.PHYSICAL)
        self.capabilities = (
            DiscretizationCapability.PROJECTION,
            DiscretizationCapability.RECONSTRUCTION,
            DiscretizationCapability.TRACE,
            DiscretizationCapability.BOUNDARY_INTEGRAL,
            DiscretizationCapability.VARIATIONAL_ASSEMBLY,
            DiscretizationCapability.SPARSE_ASSEMBLY,
            DiscretizationCapability.MATRIX_FREE,
            DiscretizationCapability.DIFFERENTIABLE_GEOMETRY,
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-element-plan",
                "mesh": mesh.mesh_id,
                "fields": [field.field_spec_id for field in field_specs],
                "precision_policy": precision.policy_id,
            }
        )

    def prepare(self, /, *, numeric_version: str = "0"):
        return FiniteElementDiscretization(self, numeric_version=numeric_version)


class FiniteElementDiscretization(AbstractPreparedDiscretization):
    mesh: CellMesh
    dof_maps: tuple[FiniteElementDofMap, ...]
    elements: tuple[tuple[FiniteElementSpec, ...], ...]
    block_geometries: tuple[tuple[FiniteElementBlockGeometry, ...], ...]
    mass_operators: tuple[SparseLinearMap, ...]
    stiffness_operators: tuple[SparseLinearMap, ...]
    cell_domain: IntegrationDomain
    exterior_facet_domain: IntegrationDomain
    interior_facet_domain: IntegrationDomain
    key: DiscretizationKey
    support: DiscreteSupport
    field_spaces: tuple[DiscreteFieldSpace, ...]
    measures: tuple[DiscreteMeasure, ...]
    precision_policy: FiniteElementPrecisionPolicy
    capabilities: tuple[DiscretizationCapability, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    numeric_version: str = eqx.field(static=True)
    preparation: PreparationReport

    def __init__(self, plan: FiniteElementPlan, /, *, numeric_version: str = "0"):
        if not isinstance(plan, FiniteElementPlan):
            raise TypeError("plan must be a FiniteElementPlan.")
        version = str(numeric_version)
        if not version:
            raise ValueError("numeric_version must be non-empty.")
        mesh = plan.mesh
        field_spaces = []
        dof_maps = []
        all_elements = []
        all_geometries = []
        mass_operators = []
        stiffness_operators = []
        cell_measures = None
        for field in plan.fields:
            elements = field.resolve(mesh)
            all_elements.append(elements)
            dof_map = FiniteElementDofMap(mesh, elements)
            dof_maps.append(dof_map)
            vector_shape = (dof_map.global_dof_count,) + field.component_shape
            vector_space = ArraySpace(vector_shape)
            layout = EntityDofLayout(
                mesh.topology.entity_sets[0].entity_set_id,
                int(mesh.coordinates.shape[0]),
                dof_map.global_dof_count,
                component_shape=field.component_shape,
            )
            field_spaces.append(
                DiscreteFieldSpace(
                    field.name,
                    mesh.support.support_id,
                    layout,
                    vector_space,
                    representation="point_value",
                    conformity="H1",
                    projection_id=canonical_fingerprint(
                        {
                            "kind": "finite-element-projection",
                            "field": field.field_spec_id,
                        }
                    ),
                    reconstruction_id=canonical_fingerprint(
                        {
                            "kind": "finite-element-reconstruction",
                            "field": field.field_spec_id,
                        }
                    ),
                )
            )
            geometries = tuple(
                _prepare_block_geometry(mesh, block, element)
                for block, element in zip(mesh.blocks, elements, strict=True)
            )
            all_geometries.append(geometries)
            mass_local = tuple(
                oe.contract(
                    "cq,qi,qj->cij",
                    geometry.physical_weights,
                    geometry.basis_values,
                    geometry.basis_values,
                )
                for geometry in geometries
            )
            stiffness_local = tuple(
                oe.contract(
                    "cq,cqid,cqjd->cij",
                    geometry.physical_weights,
                    geometry.physical_gradients,
                    geometry.physical_gradients,
                )
                for geometry in geometries
            )
            mass_operators.append(
                _assemble_local_operator(
                    dof_map,
                    mass_local,
                    "finite-element-mass",
                    positive_definite=True,
                )
            )
            stiffness_operators.append(
                _assemble_local_operator(
                    dof_map,
                    stiffness_local,
                    "finite-element-stiffness",
                    positive_definite=False,
                )
            )
            if cell_measures is None:
                cell_measures = jnp.concatenate(
                    tuple(geometry.measure for geometry in geometries), axis=0
                )
        if cell_measures is None:
            raise ValueError("Finite-element preparation produced no cell measures.")
        top_entities = mesh.topology.entity_sets[mesh.topological_dimension]
        measure_metadata = (
            DiscreteMeasure(
                "finite_element_cell_measure",
                mesh.support.support_id,
                top_entities.entity_set_id,
                cell_measures,
            ),
        )
        preparation = PreparationReport(
            capabilities=plan.capabilities,
            diagnostics=(
                "reference elements are compatible with mesh blocks",
                "cell geometry measures are positive",
                "local-to-global DOF routes are fixed",
            ),
            resource_counts={
                "vertices": int(mesh.coordinates.shape[0]),
                "cells": sum(block.cell_count for block in mesh.blocks),
                "fields": len(plan.fields),
                "global_dofs": sum(dof_map.global_dof_count for dof_map in dof_maps),
            },
        )
        spaces, measures, capabilities = validate_prepared_metadata(
            key=plan.key,
            support=mesh.support,
            field_spaces=tuple(field_spaces),
            measures=measure_metadata,
            capabilities=plan.capabilities,
            preparation=preparation,
        )
        connectivity = mesh.connectivity
        if isinstance(connectivity, PolygonalConnectivity):
            exterior = np.flatnonzero(np.asarray(connectivity.boundary_edges, dtype=bool))
            interior = np.flatnonzero(
                ~np.asarray(connectivity.boundary_edges, dtype=bool)
            )
        else:
            exterior = np.flatnonzero(np.asarray(connectivity.boundary_faces, dtype=bool))
            interior = np.flatnonzero(
                ~np.asarray(connectivity.boundary_faces, dtype=bool)
            )
        self.mesh = mesh
        self.dof_maps = tuple(dof_maps)
        self.elements = tuple(all_elements)
        self.block_geometries = tuple(all_geometries)
        self.mass_operators = tuple(mass_operators)
        self.stiffness_operators = tuple(stiffness_operators)
        self.cell_domain = IntegrationDomain(
            "cell",
            np.arange(sum(block.cell_count for block in mesh.blocks), dtype=np.int32),
            mesh.support.support_id,
        )
        self.exterior_facet_domain = IntegrationDomain(
            "exterior_facet", exterior, mesh.support.support_id
        )
        self.interior_facet_domain = IntegrationDomain(
            "interior_facet", interior, mesh.support.support_id
        )
        self.key = plan.key
        self.precision_policy = plan.precision_policy
        self.support = mesh.support
        self.field_spaces = spaces
        self.measures = measures
        self.capabilities = capabilities
        self.plan_id = plan.plan_id
        self.numeric_version = version
        self.preparation = preparation
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-finite-element",
                "plan": plan.plan_id,
                "mesh": mesh.mesh_id,
                "numeric_version": version,
            }
        )

    @property
    def mass(self) -> SparseLinearMap:
        if len(self.mass_operators) != 1:
            raise ValueError("mass is only unambiguous for one field.")
        return self.mass_operators[0]

    @property
    def stiffness(self) -> SparseLinearMap:
        if len(self.stiffness_operators) != 1:
            raise ValueError("stiffness is only unambiguous for one field.")
        return self.stiffness_operators[0]

    @property
    def vertices(self) -> Array:
        return self.mesh.coordinates

    @property
    def boundary_dof_mask(self) -> Array:
        if len(self.dof_maps) != 1:
            raise ValueError("boundary_dof_mask is only unambiguous for one field.")
        return self.dof_maps[0].boundary_dof_mask

    def reconstruct(
        self,
        field_name: str,
        coefficients: ArrayLike,
        block_name: str,
        reference_points: ArrayLike,
        /,
    ) -> Array:
        field_index = self._field_index(field_name)
        block_index = self.dof_maps[field_index].block_names.index(str(block_name))
        element = self._field_elements(field_index)[block_index]
        values, _ = element.tabulate(reference_points)
        local = jnp.asarray(coefficients)[
            self.dof_maps[field_index].cell_dofs[block_index]
        ]
        return oe.contract("qi,ci...->cq...", values, local)

    def _field_index(self, field_name: str, /) -> int:
        requested = str(field_name)
        for index, field_space in enumerate(self.field_spaces):
            if field_space.name == requested:
                return index
        raise KeyError(f"Unknown finite-element field {requested!r}.")

    def _field_elements(self, field_index: int, /) -> tuple[FiniteElementSpec, ...]:
        return self.elements[field_index]

    def evaluate_geometry(
        self,
        field_name: str,
        coordinates: ArrayLike,
        /,
    ) -> tuple[FiniteElementBlockGeometry, ...]:
        field_index = self._field_index(field_name)
        points = jnp.asarray(coordinates)
        if points.shape != self.mesh.coordinates.shape:
            raise ValueError(
                "Fixed-topology FE geometry evaluation must preserve coordinate shape."
            )
        return tuple(
            _prepare_block_geometry(
                self.mesh,
                block,
                element,
                coordinates=points,
            )
            for block, element in zip(
                self.mesh.blocks,
                self.elements[field_index],
                strict=True,
            )
        )


def _reference_rule(cell_kind: str, /) -> tuple[Array, Array]:
    axis, weights = np.polynomial.legendre.leggauss(4)
    axis = 0.5 * (axis + 1.0)
    weights = 0.5 * weights
    if cell_kind == "triangle":
        first, second = np.meshgrid(axis, axis, indexing="ij")
        points = np.stack((first, (1.0 - first) * second), axis=-1)
        combined = weights[:, None] * weights[None, :] * (1.0 - first)
        return jnp.asarray(points.reshape((-1, 2))), jnp.asarray(combined.reshape((-1,)))
    if cell_kind == "quadrilateral":
        first, second = np.meshgrid(axis, axis, indexing="ij")
        points = np.stack((first, second), axis=-1)
        combined = weights[:, None] * weights[None, :]
        return jnp.asarray(points.reshape((-1, 2))), jnp.asarray(combined.reshape((-1,)))
    if cell_kind == "tetrahedron":
        first, second, third = np.meshgrid(axis, axis, axis, indexing="ij")
        one_minus_first = 1.0 - first
        one_minus_second = 1.0 - second
        points = np.stack(
            (
                first,
                one_minus_first * second,
                one_minus_first * one_minus_second * third,
            ),
            axis=-1,
        )
        combined = (
            weights[:, None, None]
            * weights[None, :, None]
            * weights[None, None, :]
            * one_minus_first**2
            * one_minus_second
        )
        return jnp.asarray(points.reshape((-1, 3))), jnp.asarray(combined.reshape((-1,)))
    raise ValueError("Unsupported finite-element cell kind.")


def _small_inverse(metric: Array, /) -> tuple[Array, Array]:
    dimension = metric.shape[-1]
    if dimension == 2:
        a = metric[..., 0, 0]
        b = metric[..., 0, 1]
        c = metric[..., 1, 0]
        d = metric[..., 1, 1]
        determinant = a * d - b * c
        inverse = jnp.stack((d, -b, -c, a), axis=-1).reshape(metric.shape)
        return inverse / determinant[..., None, None], determinant
    if dimension == 3:
        determinant = (
            metric[..., 0, 0]
            * (
                metric[..., 1, 1] * metric[..., 2, 2]
                - metric[..., 1, 2] * metric[..., 2, 1]
            )
            - metric[..., 0, 1]
            * (
                metric[..., 1, 0] * metric[..., 2, 2]
                - metric[..., 1, 2] * metric[..., 2, 0]
            )
            + metric[..., 0, 2]
            * (
                metric[..., 1, 0] * metric[..., 2, 1]
                - metric[..., 1, 1] * metric[..., 2, 0]
            )
        )
        cofactor = jnp.empty_like(metric)
        cofactor = cofactor.at[..., 0, 0].set(
            metric[..., 1, 1] * metric[..., 2, 2] - metric[..., 1, 2] * metric[..., 2, 1]
        )
        cofactor = cofactor.at[..., 0, 1].set(
            -(
                metric[..., 1, 0] * metric[..., 2, 2]
                - metric[..., 1, 2] * metric[..., 2, 0]
            )
        )
        cofactor = cofactor.at[..., 0, 2].set(
            metric[..., 1, 0] * metric[..., 2, 1] - metric[..., 1, 1] * metric[..., 2, 0]
        )
        cofactor = cofactor.at[..., 1, 0].set(
            -(
                metric[..., 0, 1] * metric[..., 2, 2]
                - metric[..., 0, 2] * metric[..., 2, 1]
            )
        )
        cofactor = cofactor.at[..., 1, 1].set(
            metric[..., 0, 0] * metric[..., 2, 2] - metric[..., 0, 2] * metric[..., 2, 0]
        )
        cofactor = cofactor.at[..., 1, 2].set(
            -(
                metric[..., 0, 0] * metric[..., 2, 1]
                - metric[..., 0, 1] * metric[..., 2, 0]
            )
        )
        cofactor = cofactor.at[..., 2, 0].set(
            metric[..., 0, 1] * metric[..., 1, 2] - metric[..., 0, 2] * metric[..., 1, 1]
        )
        cofactor = cofactor.at[..., 2, 1].set(
            -(
                metric[..., 0, 0] * metric[..., 1, 2]
                - metric[..., 0, 2] * metric[..., 1, 0]
            )
        )
        cofactor = cofactor.at[..., 2, 2].set(
            metric[..., 0, 0] * metric[..., 1, 1] - metric[..., 0, 1] * metric[..., 1, 0]
        )
        return jnp.swapaxes(cofactor, -1, -2) / determinant[..., None, None], determinant
    raise ValueError("Only two- and three-dimensional local geometry is supported.")


def _prepare_block_geometry(
    mesh: CellMesh,
    block: CellBlock,
    element: FiniteElementSpec,
    /,
    *,
    coordinates: ArrayLike | None = None,
) -> FiniteElementBlockGeometry:
    reference_points, reference_weights = _reference_rule(block.cell_kind)
    geometry_element = lagrange_element(block.cell_kind, 1)
    geometry_values, geometry_gradients = geometry_element.tabulate(reference_points)
    basis_values, reference_gradients = element.tabulate(reference_points)
    coordinate_values = (
        mesh.coordinates if coordinates is None else jnp.asarray(coordinates)
    )
    cell_coordinates = coordinate_values[block.vertices]
    physical_points = oe.contract("qi,cid->cqd", geometry_values, cell_coordinates)
    jacobian = oe.contract("qir,cid->cqdr", geometry_gradients, cell_coordinates)
    metric = oe.contract("cqdi,cqdj->cqij", jacobian, jacobian)
    inverse_metric, determinant = _small_inverse(metric)
    measure_factor = jnp.sqrt(determinant)
    measure_factor = eqx.error_if(
        measure_factor,
        jnp.any(~jnp.isfinite(measure_factor) | (measure_factor <= 0.0)),
        "Finite-element geometry requires positive finite metric determinant.",
    )
    physical_gradients = oe.contract(
        "cqdi,cqij,qkj->cqkd",
        jacobian,
        inverse_metric,
        reference_gradients,
    )
    physical_weights = measure_factor * reference_weights[None, :]
    return FiniteElementBlockGeometry(
        block_name=block.name,
        reference_points=reference_points,
        reference_weights=reference_weights,
        basis_values=basis_values,
        reference_gradients=reference_gradients,
        physical_points=physical_points,
        physical_gradients=physical_gradients,
        physical_weights=physical_weights,
        measure=jnp.sum(physical_weights, axis=1),
    )


def _assemble_local_operator(
    dof_map: FiniteElementDofMap,
    local_values: Sequence[Array],
    kind: str,
    /,
    *,
    positive_definite: bool,
) -> SparseLinearMap:
    source_parts = []
    target_parts = []
    coefficient_parts = []
    for cell_dofs, values in zip(dof_map.cell_dofs, local_values, strict=True):
        indices = np.asarray(cell_dofs, dtype=np.int32)
        width = indices.shape[1]
        source_parts.append(
            np.broadcast_to(
                indices[:, None, :], (indices.shape[0], width, width)
            ).reshape((-1,))
        )
        target_parts.append(
            np.broadcast_to(
                indices[:, :, None], (indices.shape[0], width, width)
            ).reshape((-1,))
        )
        coefficient_parts.append(jnp.asarray(values).reshape((-1,)))
    relation = EdgeRelation(
        np.concatenate(source_parts),
        np.concatenate(target_parts),
        source_size=dof_map.global_dof_count,
        target_size=dof_map.global_dof_count,
    )
    properties = OperatorProperties(
        self_adjoint=True,
        positive_definite=positive_definite,
        positive_semidefinite=True,
        evidence={
            "self_adjoint": "construction",
            "positive_semidefinite": "construction",
            **({"positive_definite": "construction"} if positive_definite else {}),
        },
    )
    return SparseLinearMap(
        relation,
        jnp.concatenate(tuple(coefficient_parts)),
        properties=properties,
        operator_id=canonical_fingerprint({"kind": kind, "dof_map": dof_map.dof_map_id}),
    )


__all__ = [
    "FiniteElementDiscretization",
    "FiniteElementDofMap",
    "FiniteElementFieldSpec",
    "FiniteElementPlan",
    "IntegrationDomain",
]
