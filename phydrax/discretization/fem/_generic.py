#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, final, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike, DTypeLike

import phydrax.ein as ein
from phydrax.discretization._reference_cell_types import (
    CellBlock,
    CellGeometryElement,
    PolyhedralBlock,
)
from phydrax.discretization.fem._reference_types import FiniteElementSpec

from ... import typing as typing_owner
from ..._coordinate_measure import CoordinateFrame, CoordinateMeasurePlan
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...ein import contract
from ...exterior._algebra import map_reference_values, pullback, vector_to_form
from ...exterior._form_type import FormType, FormValueSpec
from ...linalg import (
    ArraySpace,
    BlockSpace,
    OperatorProperties,
)
from ...sparse import (
    EdgeRelation,
    gather_routes,
    route_reduce,
    RowRelation,
    SparseLinearMap,
)
from ...typing import checked
from .. import _cell_geometry as cell_geometry_owner, _cell_mesh as cell_mesh_owner
from .._adaptive_simplex import MaskedSimplexMesh
from .._cell_complex import (
    IntervalConnectivity,
    PolygonalConnectivity,
    PolyhedralConnectivity,
    TetrahedralConnectivity,
)
from .._coordinate_budget import CoordinateEnclosureBudget
from .._core import (
    DiscretizationCapability,
    DiscretizationKey,
    DiscretizationRole,
    PreparationReport,
)
from .._hexahedral import HexahedralConnectivity
from .._integration_domain import IntegrationDomain
from .._lifecycle import (
    AbstractDiscretizationPlan,
    validate_prepared_metadata,
)
from .._local_variational import (
    AbstractPreparedLocalDiscretization,
    LocalFieldBinding,
    LocalVariationalCapabilities,
    PreparedLocalRegion,
)
from .._measure import DiscreteMeasure
from .._side_actions import (
    FacetTraceRule,
    PreparedTraceAction,
    SideTraceQuantity,
)
from .._spaces import BlockDofLayout, DiscreteFieldSpace, EntityDofLayout
from .._support import DiscreteSupport
from .._topology import EntitySelection
from .._views import FieldTraceSide
from . import (
    _form_elements as form_elements_owner,
    _local_provider as local_provider_owner,
)
from ._assembly import (
    _PreparedCoordinateMaps,
    assemble_local_operator,
    build_finite_element_dof_coordinates,
    build_finite_element_dof_routes,
    canonical_finite_element_routes,
    evaluate_coordinate_map,
    FormDofLayout,
    geometry_local_tensors,
    lifted_finite_element_dof_layout,
    linear_reference_element,
    local_mass_tensor,
    local_stiffness_tensor,
    NodalDofLayout,
    prepare_local_assembly_relation,
    quotient_cell_transforms,
    quotient_dof_layout,
    quotient_dof_routes,
    QuotientDofLayout,
    SingleAssociationDofLayout,
    validate_restored_fem_state,
)
from ._precision import FiniteElementPrecisionPolicy
from ._reference import discontinuous_element, lagrange_element
from ._reference_rules import degree_aware_reference_rule
from ._restoration import authenticate_restored_node, RestorationValidationContext
from ._source_ownership import AbstractFiniteElementSourceOwnership


if TYPE_CHECKING:
    from ._cell_map import PreparedFiniteElementCellMap


def _field_element_assignments(
    elements: FiniteElementSpec | Mapping[str, FiniteElementSpec],
    block_names: Sequence[str] | None,
    /,
) -> tuple[tuple[str, ...], tuple[FiniteElementSpec, ...]]:
    if isinstance(elements, FiniteElementSpec):
        names = () if block_names is None else tuple(str(block) for block in block_names)
        element_values = (elements,) if not names else (elements,) * len(names)
    else:
        items = tuple(
            sorted((str(block), element) for block, element in elements.items())
        )
        if not items:
            raise ValueError(
                "FiniteElementFieldSpec.elements must be nonempty; got an empty mapping"
            )
        names = tuple(block for block, _ in items)
        element_values = tuple(element for _, element in items)
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError(
            "FiniteElementFieldSpec.block_names must be unique and nonempty; got invalid block names"
        )
    if not all(isinstance(element, FiniteElementSpec) for element in element_values):
        raise TypeError(
            "FiniteElementFieldSpec.elements must contain FiniteElementSpec instances; got an incompatible element"
        )
    return names, element_values


def _resolve_field_assignments(
    mesh: cell_mesh_owner.CellMesh,
    names: tuple[str, ...],
    elements: tuple[FiniteElementSpec, ...],
    /,
) -> tuple[FiniteElementSpec, ...]:
    if not names:
        if len(elements) != 1:
            raise ValueError(
                "FiniteElementFieldSpec.elements must contain one implicit element; got a different element count"
            )
        return (elements[0],) * len(mesh.blocks)
    assignments = dict(zip(names, elements, strict=True))
    if set(assignments) != {block.name for block in mesh.blocks}:
        raise ValueError(
            "FiniteElementFieldSpec.block_names must match the mesh block names; got different assignments"
        )
    return tuple(assignments[block.name] for block in mesh.blocks)


def _validate_resolved_elements(
    mesh: cell_mesh_owner.CellMesh, elements: tuple[FiniteElementSpec, ...], /
) -> None:
    for block, element in zip(mesh.blocks, elements, strict=True):
        if block.cell_kind != element.cell_kind:
            raise ValueError(
                f"FiniteElementFieldSpec.elements.cell_kind must match block {block.name!r} ({block.cell_kind!r}); got {element.cell_kind!r}"
            )
    if len({element.conformity for element in elements}) != 1:
        raise ValueError(
            "FiniteElementFieldSpec.elements.conformity must agree across mesh blocks; got mixed conformities"
        )
    representations = {element.representation for element in elements}
    canonical_moments = all(
        element.form_basis is not None
        and element.representation in ("polynomial_moment", "rational_moment")
        for element in elements
    )
    if len(representations) != 1 and not canonical_moments:
        raise ValueError(
            "FiniteElementFieldSpec.elements.representation must agree across mesh blocks; got mixed coefficient representations"
        )
    if canonical_moments and any(
        element.value_spec != elements[0].value_spec for element in elements
    ):
        raise ValueError(
            "FiniteElementFieldSpec.elements.value_spec must agree on form degree, twist, proxy, and component identity; got incompatible moment specifications"
        )
    if len({element.mapping for element in elements}) != 1:
        raise ValueError(
            "FiniteElementFieldSpec.elements.mapping must agree across mesh blocks; got mixed mappings"
        )
    if len({element.value_shape for element in elements}) != 1:
        raise ValueError(
            "FiniteElementFieldSpec.elements.value_shape must agree across mesh blocks; got mixed value shapes"
        )


@final
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
    ) -> None:
        field_name = str(name)
        if not field_name:
            raise ValueError(
                "FiniteElementFieldSpec.name must be nonempty; got an empty name"
            )
        components = tuple(component_shape)
        if any(size <= 0 for size in components):
            raise ValueError(
                "FiniteElementFieldSpec.component_shape must have positive dimensions; got a nonpositive component dimension"
            )
        names, element_values = _field_element_assignments(elements, block_names)
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

    def resolve(self, mesh: cell_mesh_owner.CellMesh, /) -> tuple[FiniteElementSpec, ...]:
        resolved = _resolve_field_assignments(mesh, self.block_names, self.elements)
        _validate_resolved_elements(mesh, resolved)
        return resolved


@final
class FiniteElementDofSourceProjection(StrictModule, NonTrainableState):
    """Provide source-frame routes and identities for resident DOF construction."""

    source: FiniteElementDofMap
    cell_rows: tuple[Array, ...]
    block_indices: Array
    cell_frames: tuple[Array, ...]
    source_keys: tuple[str, ...] = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    source_authority: AbstractFiniteElementSourceOwnership

    @checked
    def __init__(
        self,
        source: FiniteElementDofMap,
        cell_rows: Sequence[ArrayLike],
        block_indices: ArrayLike,
        cell_frames: Sequence[ArrayLike],
        source_keys: Sequence[str],
        source_id: str,
        source_authority: AbstractFiniteElementSourceOwnership,
        /,
    ) -> None:
        if source_id != source_authority.plan_id:
            raise ValueError(
                "FiniteElementDofSourceProjection.source_id must match the source operation; "
                f"got {source_id!r}, expected {source_authority.plan_id!r}"
            )
        validate_restored_fem_state(source, source_authority.source_dof_map)
        actual_keys = source_authority.source_dof_identity_keys(source)
        if tuple(source_keys) != actual_keys:
            raise ValueError(
                "FiniteElementDofSourceProjection.source_keys must match authored node/moment identities; "
                "got different keys"
            )
        indices = np.asarray(block_indices)
        rows = tuple(np.asarray(value) for value in cell_rows)
        if not np.issubdtype(indices.dtype, np.integer) or any(
            not np.issubdtype(value.dtype, np.integer) for value in rows
        ):
            raise TypeError(
                "FiniteElementDofSourceProjection.cell_rows_block_indices must contain integer arrays; got a noninteger dtype"
            )
        frames = tuple(np.asarray(value) for value in cell_frames)
        keys = tuple(source_keys)
        if (
            indices.shape != (len(rows),)
            or len(frames) != len(rows)
            or np.any(indices < 0)
            or np.any(indices >= len(source.cell_dofs))
            or len(keys) != source.global_dof_count
            or len(set(keys)) != len(keys)
            or any(not isinstance(key, str) or not key for key in keys)
            or not isinstance(source_id, str)
            or not source_id
        ):
            raise ValueError(
                "FiniteElementDofSourceProjection.inputs must cover source identities and frames; "
                "got invalid block indices, rows, frames, keys, or source_id"
            )
        for index, cells, maps in zip(indices, rows, frames, strict=True):
            routes = source.cell_dofs[int(index)]
            if (
                cells.ndim != 1
                or np.any(cells < 0)
                or np.any(cells >= routes.shape[0])
                or maps.ndim != 4
                or maps.shape[:2] != (cells.size, routes.shape[1])
                or maps.shape[2] != maps.shape[3]
                or maps.shape[2] < 2
                or np.any(~np.isfinite(maps))
            ):
                raise ValueError(
                    "FiniteElementDofSourceProjection.cell_rows_frames must cover valid cell routes with finite homogeneous frames; got invalid routes or frames"
                )
            if not np.array_equal(
                maps[..., -1, :],
                np.broadcast_to(
                    np.eye(maps.shape[-1])[-1], maps.shape[:-2] + (maps.shape[-1],)
                ),
            ):
                raise ValueError(
                    "FiniteElementDofSourceProjection.cell_frames must be homogeneous affine maps; got an invalid homogeneous row"
                )
            lifted_rows = source_authority.source_lifted_routes[int(index)][
                jnp.asarray(cells)
            ]
            actual_frames = source_authority.source_lifted_frames[lifted_rows]
            if array_tree_fingerprint(maps) != array_tree_fingerprint(actual_frames):
                raise ValueError(
                    "FiniteElementDofSourceProjection.cell_frames must match authored orbit frames; "
                    f"got different frames for source block {int(index)}"
                )
        self.source = source
        self.cell_rows = tuple(jnp.asarray(value) for value in rows)
        self.block_indices = jnp.asarray(indices)
        self.cell_frames = tuple(jnp.asarray(value) for value in frames)
        self.source_keys = keys
        self.source_id = str(source_id)
        self.source_authority = source_authority

    @authenticate_restored_node
    def validate_restored(
        self, /, *, validation_context: RestorationValidationContext
    ) -> None:
        self.source_authority.validate_restored(validation_context=validation_context)
        self.source.validate_restored(validation_context=validation_context)
        rebuilt = FiniteElementDofSourceProjection(
            self.source,
            self.cell_rows,
            self.block_indices,
            self.cell_frames,
            self.source_keys,
            self.source_id,
            self.source_authority,
        )
        validate_restored_fem_state(self, rebuilt)


class _ProjectedFiniteElementDofs(StrictModule):
    routes: tuple[Array, ...]
    relations: tuple[RowRelation, ...]
    transforms: tuple[Array, ...]
    orientations: tuple[Array, ...]
    coordinates: np.ndarray
    boundary: np.ndarray
    scientific_keys: tuple[str, ...] = eqx.field(static=True)
    coordinate_gather: Array | None
    coordinate_frames: Array | None


def _project_finite_element_dof_routes(
    mesh: cell_mesh_owner.CellMesh,
    resolved: tuple[FiniteElementSpec, ...],
    components: tuple[int, ...],
    source_projection: FiniteElementDofSourceProjection,
    global_count: int,
    coordinate_gather: Array | None,
    /,
) -> _ProjectedFiniteElementDofs:
    """Project source coefficient routes and authored coordinate frames to storage."""
    source = source_projection.source
    if source.construction_projection is not None:
        raise ValueError(
            "FiniteElementDofMap.source_projection.source must be a prepartition DOF map; "
            "got a source with another projection"
        )
    if (
        mesh.storage is None
        or mesh.storage.logical_topology_id != source.construction_mesh.topology_id
    ):
        raise ValueError(
            "FiniteElementDofMap.mesh.storage must bind the source topology; "
            "got missing storage or a different topology"
        )
    indices = np.asarray(source_projection.block_indices)
    if len(indices) != len(mesh.blocks) or source.component_shape != components:
        raise ValueError(
            "FiniteElementDofMap.source_projection must cover every resident field block with matching components; got incompatible block indices or components"
        )
    source_routes, source_transforms, source_signs = [], [], []
    for block, element, index, rows, frames in zip(
        mesh.blocks,
        resolved,
        indices,
        source_projection.cell_rows,
        source_projection.cell_frames,
        strict=True,
    ):
        if (
            source.block_names[int(index)] != block.name
            or source.element_ids[int(index)] != element.element_id
            or rows.shape != (block.cell_count,)
            or source.cell_dofs[int(index)].shape[1] != element.local_dof_count
            or frames.shape[2:]
            != (mesh.ambient_dimension + 1, mesh.ambient_dimension + 1)
        ):
            raise ValueError(
                "FiniteElementDofMap.source_projection must preserve resident reference metadata; "
                f"got an incompatible block {block.name!r}"
            )
        source_block = source.construction_mesh.blocks[int(index)]
        cell_rows = np.asarray(rows)
        if not np.array_equal(
            np.asarray(block.global_ids),
            np.asarray(source_block.global_ids)[cell_rows],
        ) or not np.array_equal(
            np.asarray(mesh.vertex_global_ids)[np.asarray(block.vertices)],
            np.asarray(source.construction_mesh.vertex_global_ids)[
                np.asarray(source_block.vertices)[cell_rows]
            ],
        ):
            raise ValueError(
                "FiniteElementDofMap.source_projection must preserve cell IDs and ordered corners; "
                f"got different IDs or corners for block {block.name!r}"
            )
        source_routes.append(np.asarray(source.cell_dofs[int(index)])[np.asarray(rows)])
        source_transforms.append(source.cell_transforms[int(index)][rows])
        source_signs.append(source.orientations[int(index)][rows])
    resident = np.unique(np.concatenate([rows.reshape(-1) for rows in source_routes]))
    if resident.size != global_count:
        raise ValueError(
            "FiniteElementDofMap.source_projection must preserve source DOF orbits; "
            f"got {resident.size} resident DOFs, expected {global_count}"
        )
    block_dofs = tuple(
        jnp.asarray(np.searchsorted(resident, rows), dtype=jnp.int32)
        for rows in source_routes
    )
    relations = tuple(RowRelation(rows, source_size=global_count) for rows in block_dofs)
    transforms, orientations = tuple(source_transforms), tuple(source_signs)
    dof_coordinates = np.asarray(source.dof_coordinates)[resident]
    boundary = np.asarray(source.boundary_dof_mask)[resident]
    scientific_keys = tuple(source_projection.source_keys[int(row)] for row in resident)
    coordinate_frames = None
    if mesh.periodic_topology is not None:
        gather = np.full((global_count,), -1, dtype=np.int32)
        coordinate_frames_host = np.empty(
            (global_count, mesh.ambient_dimension + 1, mesh.ambient_dimension + 1)
        )
        offset = 0
        for rows, frames in zip(block_dofs, source_projection.cell_frames, strict=True):
            flat_frames = np.asarray(frames).reshape(
                (-1, mesh.ambient_dimension + 1, mesh.ambient_dimension + 1)
            )
            for position, row in enumerate(np.asarray(rows).reshape(-1)):
                if gather[row] < 0:
                    gather[row] = offset + position
                    coordinate_frames_host[row] = flat_frames[position]
            offset += rows.size
        coordinate_gather = jnp.asarray(gather)
        coordinate_frames = jnp.asarray(coordinate_frames_host)
    return _ProjectedFiniteElementDofs(
        block_dofs,
        relations,
        transforms,
        orientations,
        dof_coordinates,
        boundary,
        scientific_keys,
        coordinate_gather,
        coordinate_frames,
    )


@final
class FiniteElementDofMap(StrictModule, NonTrainableState):
    """Per-block FE local gathers into one global field coordinate array.

    ``cell_transforms`` map shared entity coefficients to local moments for
    every polynomial order. Geometry retains the untransformed reference basis;
    assembly, projection and point evaluation each apply this map exactly once.

    On a periodic mesh, entity DOFs are numbered on the quotient complex
    (``quotient_*`` associations): seam copies of one quotient entity gather
    the same DOFs, with the orientation signs of the periodic topology.
    ``coordinate_gather`` then locates each quotient DOF's coordinate
    among the flattened (block, cell, local DOF) mapped nodes of its
    representative lifted DOF; it is ``None`` on meshes without quotient DOFs.
    """

    block_names: tuple[str, ...] = eqx.field(static=True)
    element_ids: tuple[str, ...] = eqx.field(static=True)
    cell_dofs: tuple[Array, ...]
    relations: tuple[RowRelation, ...]
    assembly_relation: EdgeRelation
    orientations: tuple[Array, ...]
    cell_transforms: tuple[Array, ...]
    coordinate_gather: Array | None
    coordinate_frames: Array | None
    cell_coordinate_weights: tuple[Array, ...]
    coordinate_dofs: tuple[Array, ...]
    coordinate_count: int = eqx.field(static=True)
    require_consistent_coordinates: bool = eqx.field(static=True)
    global_dof_count: int = eqx.field(static=True)
    entity_dof_counts: tuple[int, ...] = eqx.field(static=True)
    entity_dofs_per_entity: tuple[int, ...] = eqx.field(static=True)
    component_shape: tuple[int, ...] = eqx.field(static=True)
    association: str = eqx.field(static=True)
    boundary_dof_mask: Array
    dof_coordinates: Array
    dof_map_id: str = eqx.field(static=True)
    scientific_keys: tuple[str, ...] = eqx.field(static=True)
    source_projection_id: str | None = eqx.field(static=True)
    construction_mesh: cell_mesh_owner.CellMesh
    construction_elements: tuple[FiniteElementSpec, ...]
    construction_geometry: cell_geometry_owner.CellGeometrySpec
    construction_projection: FiniteElementDofSourceProjection | None

    @checked
    def __init__(
        self,
        mesh: cell_mesh_owner.CellMesh,
        elements: Sequence[FiniteElementSpec],
        /,
        *,
        component_shape: Sequence[int] = (),
        coordinate_spec: cell_geometry_owner.CellGeometrySpec | None = None,
        source_projection: FiniteElementDofSourceProjection | None = None,
        budget: CoordinateEnclosureBudget | None = None,
        _coordinate_preparation: _PreparedCoordinateMaps | None = None,
    ) -> None:
        resolved = tuple(elements)
        if len(resolved) != len(mesh.blocks):
            raise ValueError(
                "FiniteElementDofMap.elements must contain one finite element per mesh block; got a different element count"
            )
        components = tuple(component_shape)
        if any(size <= 0 for size in components):
            raise ValueError(
                "FiniteElementDofMap.component_shape must have positive dimensions; got a nonpositive component dimension"
            )
        geometry = coordinate_spec
        if geometry is None:
            geometry = (
                cell_geometry_owner.CellGeometrySpec.affine(mesh)
                if mesh.storage is None
                else mesh.storage.restore_geometry()
            )
        routing = lifted_finite_element_dof_layout(mesh, resolved, components)
        block_dofs, orientations, relations = build_finite_element_dof_routes(
            mesh, resolved, routing
        )

        coordinate_weights, coordinate_dofs, boundary, dof_coordinates = (
            build_finite_element_dof_coordinates(
                mesh,
                resolved,
                routing,
                block_dofs,
                geometry,
                budget=budget,
                coordinate_preparation=_coordinate_preparation,
            )
        )
        layout = (
            routing
            if mesh.periodic_topology is None or routing.association == "cell"
            else quotient_dof_layout(mesh, routing, mesh.periodic_topology)
        )
        association = layout.association
        global_count = layout.global_count
        entity_dof_counts = layout.entity_dof_counts
        entity_dofs_per_entity = layout.entity_dofs_per_entity
        match routing:
            case FormDofLayout():
                transforms = tuple(
                    jax.device_put(value) for value in routing.canonical_transforms
                )
            case NodalDofLayout() | SingleAssociationDofLayout():
                transforms = tuple(
                    jax.device_put(
                        np.eye(routes.shape[1], dtype=np.float64)[None]
                        * np.asarray(signs)[:, None, :]
                    )
                    for routes, signs in zip(block_dofs, orientations, strict=True)
                )
            case _:
                raise TypeError(
                    "FiniteElementDofMap.routing must be a lifted DOF layout; "
                    f"got {type(routing).__name__}"
                )
        coordinate_gather = None
        match layout:
            case QuotientDofLayout():
                transforms = quotient_cell_transforms(
                    layout.quotient, block_dofs, transforms
                )
                transforms = tuple(
                    jax.device_put(
                        np.asarray(value)
                        * layout.quotient.signs[np.asarray(routes)][:, None, :]
                    )
                    for value, routes in zip(transforms, block_dofs, strict=True)
                )
                block_dofs, orientations, relations, gather = quotient_dof_routes(
                    layout.quotient, global_count, block_dofs, orientations
                )
                boundary = layout.quotient.boundary
                # A quotient DOF sits at its representative lifted DOF.
                dof_coordinates = np.asarray(dof_coordinates)[
                    layout.quotient.representatives
                ]
                coordinate_gather = jax.device_put(np.asarray(gather, dtype=np.int32))
            case FormDofLayout() | NodalDofLayout() | SingleAssociationDofLayout():
                pass
            case _:
                raise TypeError(
                    "FiniteElementDofMap.layout must describe lifted or quotient DOFs; "
                    f"got {type(layout).__name__}"
                )
        coordinate_frames = None
        scientific_keys: tuple[str, ...] = ()
        projection_identity = None
        if source_projection is not None:
            projected = _project_finite_element_dof_routes(
                mesh,
                resolved,
                components,
                source_projection,
                global_count,
                coordinate_gather,
            )
            block_dofs = projected.routes
            relations = projected.relations
            transforms = projected.transforms
            orientations = projected.orientations
            dof_coordinates = projected.coordinates
            boundary = projected.boundary
            scientific_keys = projected.scientific_keys
            coordinate_gather = projected.coordinate_gather
            coordinate_frames = projected.coordinate_frames
            projection_identity = {
                "source": source_projection.source_id,
                "source_dof_map": source_projection.source.dof_map_id,
                "scientific_keys": scientific_keys,
                "coordinate_frames": None
                if coordinate_frames is None
                else array_tree_fingerprint(coordinate_frames),
            }

        canonical_routes, canonical_orientations = canonical_finite_element_routes(
            mesh, block_dofs, orientations
        )
        self.block_names = tuple(block.name for block in mesh.blocks)
        self.element_ids = tuple(element.element_id for element in resolved)
        self.cell_dofs = tuple(block_dofs)
        self.orientations = tuple(orientations)
        self.cell_transforms = transforms
        self.coordinate_gather = coordinate_gather
        self.coordinate_frames = coordinate_frames
        self.scientific_keys = scientific_keys
        self.source_projection_id = (
            None if source_projection is None else source_projection.source_id
        )
        self.construction_mesh = mesh
        self.construction_elements = resolved
        self.construction_geometry = geometry
        self.construction_projection = source_projection
        self.cell_coordinate_weights = tuple(
            jnp.asarray(value) for value in coordinate_weights
        )
        self.coordinate_dofs = coordinate_dofs
        self.coordinate_count = geometry.coordinates.shape[0]
        self.require_consistent_coordinates = all(
            element.conformity == "H1" and element.representation == "point_value"
            for element in resolved
        )
        self.relations = tuple(relations)
        self.global_dof_count = global_count
        self.entity_dof_counts = entity_dof_counts
        self.entity_dofs_per_entity = entity_dofs_per_entity
        self.component_shape = components
        self.assembly_relation = prepare_local_assembly_relation(
            block_dofs, global_count, components
        )
        self.association = association
        self.boundary_dof_mask = jnp.asarray(boundary)
        self.dof_coordinates = jax.device_put(dof_coordinates)
        self.dof_map_id = canonical_fingerprint(
            {
                "kind": "finite-element-dof-map",
                "mesh": mesh.topology_id,
                "elements": [element.element_id for element in resolved],
                "global_dof_count": global_count,
                "entity_dof_counts": list(entity_dof_counts),
                "entity_dofs_per_entity": list(entity_dofs_per_entity),
                "component_shape": list(components),
                "association": association,
                "cell_dofs": [
                    array_tree_fingerprint(value) for value in canonical_routes
                ],
                "orientations": [
                    array_tree_fingerprint(value) for value in canonical_orientations
                ],
                "cell_transforms": [
                    array_tree_fingerprint(
                        np.asarray(value)[
                            np.argsort(np.asarray(block.global_ids), kind="stable")
                        ]
                    )
                    for block, value in zip(mesh.blocks, transforms, strict=True)
                ],
                "cell_coordinate_weights": [
                    array_tree_fingerprint(np.asarray(value))
                    for value in coordinate_weights
                ],
                "coordinate_layout": geometry.geometry_layout_id,
                "source_projection": projection_identity,
            }
        )

    @authenticate_restored_node
    def validate_restored(
        self,
        /,
        *,
        validation_context: RestorationValidationContext,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> None:
        """Rebuild every route/frame/ID from retained independent source inputs."""
        if self.construction_projection is not None:
            self.construction_projection.validate_restored(
                validation_context=validation_context
            )
        rebuilt = FiniteElementDofMap(
            self.construction_mesh,
            self.construction_elements,
            component_shape=self.component_shape,
            coordinate_spec=self.construction_geometry,
            source_projection=self.construction_projection,
            budget=budget,
        )
        validate_restored_fem_state(self, rebuilt)

    def evaluate_coordinates(
        self,
        mesh: cell_mesh_owner.CellMesh,
        coordinates: ArrayLike,
        /,
    ) -> Array:
        """Evaluate reference dual points using the coordinate coefficient bank."""
        points = jnp.asarray(coordinates)
        if points.shape != (self.coordinate_count, mesh.ambient_dimension):
            raise ValueError(
                "FiniteElementDofMap.coordinates must preserve coordinate-map shape; got incompatible shape"
            )
        if self.coordinate_gather is not None:
            mapped_blocks = [
                (weights @ points[geometry_dofs]).reshape((-1, mesh.ambient_dimension))
                for weights, geometry_dofs in zip(
                    self.cell_coordinate_weights, self.coordinate_dofs, strict=True
                )
            ]
            mapped = jnp.concatenate(mapped_blocks, axis=0)[self.coordinate_gather]
            if self.coordinate_frames is not None:
                mapped = (
                    contract("nij,nj->ni", self.coordinate_frames[:, :-1, :-1], mapped)
                    + self.coordinate_frames[:, :-1, -1]
                )
            return mapped
        accumulated = jnp.zeros(
            (self.global_dof_count, mesh.ambient_dimension), dtype=points.dtype
        )
        counts = jnp.zeros((self.global_dof_count,), dtype=jnp.int32)
        lower = (
            jnp.full_like(accumulated, jnp.inf)
            if self.require_consistent_coordinates
            else None
        )
        upper = (
            jnp.full_like(accumulated, -jnp.inf)
            if self.require_consistent_coordinates
            else None
        )
        for weights, geometry_dofs, field_dofs in zip(
            self.cell_coordinate_weights,
            self.coordinate_dofs,
            self.cell_dofs,
            strict=True,
        ):
            mapped = weights @ points[geometry_dofs]
            accumulated = accumulated.at[field_dofs].add(mapped)
            counts = counts.at[field_dofs].add(1)
            if lower is not None and upper is not None:
                lower = lower.at[field_dofs].min(mapped)
                upper = upper.at[field_dofs].max(mapped)
        result = accumulated / counts[:, None]
        if lower is not None and upper is not None:
            tolerance = (
                1024
                * jnp.finfo(points.dtype).eps
                * jnp.maximum(1.0, jnp.max(jnp.abs(points)))
            )
            result = eqx.error_if(
                result,
                jnp.max(upper - lower) > tolerance,
                "Shared H1 nodes have inconsistent mapped physical coordinates.",
            )
        return result


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
    coordinate_measure: CoordinateMeasurePlan | None = None


class FiniteElementRuntimeData(StrictModule, NonTrainableState):
    """Dynamic fixed-topology geometry realization for FE execution."""

    coordinates: Array
    topology_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)
    numeric_version: str = eqx.field(static=True)
    runtime_id: str = eqx.field(static=True)
    coordinate_frame: CoordinateFrame = eqx.field(static=True, default="reference")

    @checked
    def __init__(
        self,
        mesh: cell_mesh_owner.CellMesh,
        coordinates: ArrayLike,
        /,
        *,
        numeric_version: str,
        geometry_layout_id: str | None = None,
        coordinate_frame: CoordinateFrame = "reference",
    ) -> None:
        points = jnp.asarray(coordinates)
        if points.ndim != 2 or points.shape[1] != mesh.ambient_dimension:
            raise ValueError(
                "FiniteElementRuntimeData.coordinates must preserve ambient dimension; got incompatible shape"
            )
        version = str(numeric_version)
        if not version:
            raise ValueError(
                "finite_element.numeric_version must be nonempty; got an empty revision"
            )
        self.coordinates = points

        self.coordinate_frame = typing_owner.parse(
            coordinate_frame, CoordinateFrame, "coordinate_frame"
        )
        self.topology_id = mesh.topology_id
        layout_id = (
            mesh.geometry_layout_id
            if geometry_layout_id is None
            else str(geometry_layout_id)
        )
        if not layout_id:
            raise ValueError(
                "FiniteElementRuntimeData.geometry_layout_id must be nonempty; got an empty identifier"
            )
        self.geometry_layout_id = layout_id
        self.numeric_version = version
        self.runtime_id = canonical_fingerprint(
            {
                "kind": "finite-element-runtime",
                "topology": mesh.topology_id,
                "geometry_layout": layout_id,
                "numeric_version": version,
                **(
                    {"coordinate_frame": self.coordinate_frame}
                    if self.coordinate_frame != "reference"
                    else {}
                ),
            }
        )


def _facet_routes(mesh: cell_mesh_owner.CellMesh, /) -> tuple[np.ndarray, ...]:
    connectivity = mesh.connectivity
    match connectivity:
        case IntervalConnectivity():
            cell_facets = np.asarray(connectivity.cell_vertices, dtype=np.int32)
            valid = np.ones_like(cell_facets, dtype=np.bool_)
            facet_count = connectivity.vertex_count
        case PolygonalConnectivity():
            cell_facets = np.asarray(connectivity.cell_edges, dtype=np.int32)
            valid = np.asarray(connectivity.cell_edge_valid, dtype=np.bool_)
            facet_count = connectivity.edges.shape[0]
        case PolyhedralConnectivity():
            return (
                np.asarray(connectivity.face_owner, dtype=np.int32),
                np.asarray(connectivity.face_neighbor, dtype=np.int32),
                np.asarray(connectivity.face_owner_local, dtype=np.int32),
                np.asarray(connectivity.face_neighbor_local, dtype=np.int32),
            )
        case cell_mesh_owner.SimplicialConnectivity():
            cell_facets = np.asarray(
                connectivity.cell_entities[mesh.topological_dimension - 1]
            )
            valid = np.ones_like(cell_facets, dtype=np.bool_)
            facet_count = connectivity.entities[mesh.topological_dimension - 1].shape[0]
        case TetrahedralConnectivity() | HexahedralConnectivity():
            cell_facets = np.asarray(connectivity.cell_faces, dtype=np.int32)
            valid = np.ones_like(cell_facets, dtype=np.bool_)
            facet_count = connectivity.faces.shape[0]
        case _:
            raise TypeError(
                "finite_element.mesh.connectivity must provide supported facet incidence; "
                f"got {type(connectivity).__name__}"
            )
    owner = np.full((facet_count,), -1, dtype=np.int32)
    neighbor = np.full((facet_count,), -1, dtype=np.int32)
    owner_local = np.full((facet_count,), -1, dtype=np.int32)
    neighbor_local = np.full((facet_count,), -1, dtype=np.int32)
    for cell in range(cell_facets.shape[0]):
        for local in range(cell_facets.shape[1]):
            if not valid[cell, local]:
                continue
            facet = int(cell_facets[cell, local])
            if owner[facet] < 0:
                owner[facet] = cell
                owner_local[facet] = local
            else:
                # Owner/neighbor facet pairing is a manifold contract; curve
                # network junctions (valence >= 3) have no single neighbor.
                if neighbor[facet] >= 0:
                    raise ValueError(
                        "finite_element.mesh.facets must be shared by at most two cells; "
                        "got a junction with more than two incident cells"
                    )
                neighbor[facet] = cell
                neighbor_local[facet] = local
    if np.any(owner < 0):
        raise ValueError(
            "finite_element.mesh.facets must have owner cells; got an unowned facet"
        )
    return owner, neighbor, owner_local, neighbor_local


def _host_metric_determinant(jacobian: np.ndarray, /) -> np.ndarray:
    """Validate host geometry without squaring a square Jacobian's condition."""
    if jacobian.shape[-2] == jacobian.shape[-1]:
        return np.asarray(np.linalg.det(jacobian) ** 2)
    return np.asarray(np.linalg.det(np.swapaxes(jacobian, -1, -2) @ jacobian))


def _cell_metric_determinants(
    cell_kind: str, points: np.ndarray, /
) -> tuple[np.ndarray, str]:
    """Evaluate the admitted cell family's metric determinant on host geometry."""
    if cell_kind.startswith("simplex:"):
        edge_matrix = np.swapaxes(points[:, 1:] - points[:, :1], -1, -2)
        determinant = _host_metric_determinant(edge_matrix)
        return (
            np.asarray(determinant),
            "finite_element.mesh.simplex_metric must have positive finite metric determinant; got a nonpositive or nonfinite determinant",
        )
    if cell_kind.startswith("tensor:"):
        chart = linear_reference_element(cell_kind)
        gradients = np.asarray(chart.tabulate(chart.reference_nodes)[1])
        jacobian = np.asarray(ein.contract("qir,cid->cqdr", gradients, points))
        determinant = _host_metric_determinant(jacobian)
        return (
            np.asarray(determinant),
            "finite_element.mesh.tensor_metric must have positive finite metric determinant; got a nonpositive or nonfinite determinant",
        )
    if cell_kind == "interval":
        difference = points[:, 1] - points[:, 0]
        determinant = np.sum(difference * difference, axis=-1)
    elif cell_kind == "triangle":
        first = points[:, 1] - points[:, 0]
        second = points[:, 2] - points[:, 0]
        determinant = _host_metric_determinant(np.stack((first, second), axis=-1))
    elif cell_kind == "quadrilateral":
        first = points[:, 1] - points[:, 0]
        second = points[:, 3] - points[:, 0]
        determinant = _host_metric_determinant(np.stack((first, second), axis=-1))
    elif cell_kind == "tetrahedron":
        edge_matrix = np.stack(
            (
                points[:, 1] - points[:, 0],
                points[:, 2] - points[:, 0],
                points[:, 3] - points[:, 0],
            ),
            axis=-1,
        )
        determinant = _host_metric_determinant(edge_matrix)
    elif cell_kind == "hexahedron":
        edge_matrix = np.stack(
            (
                points[:, 1] - points[:, 0],
                points[:, 3] - points[:, 0],
                points[:, 4] - points[:, 0],
            ),
            axis=-1,
        )
        determinant = _host_metric_determinant(edge_matrix)
    elif cell_kind in ("prism", "pyramid"):
        tetrahedra = (
            ((0, 1, 2, 3), (1, 2, 4, 3), (2, 4, 5, 3))
            if cell_kind == "prism"
            else ((0, 1, 2, 4), (0, 2, 3, 4))
        )
        determinants = []
        for first_vertex, second_vertex, third_vertex, fourth_vertex in tetrahedra:
            edge_matrix = np.stack(
                (
                    points[:, second_vertex] - points[:, first_vertex],
                    points[:, third_vertex] - points[:, first_vertex],
                    points[:, fourth_vertex] - points[:, first_vertex],
                ),
                axis=-1,
            )
            determinants.append(_host_metric_determinant(edge_matrix))
        determinant = np.min(np.stack(tuple(determinants), axis=-1), axis=-1)
    else:
        raise ValueError(
            "finite_element.mesh.cell_kind must select a supported reference cell; got an unsupported cell kind"
        )
    return (
        np.asarray(determinant),
        "finite_element.mesh.cell_metric must have positive finite metric determinant; got a nonpositive or nonfinite determinant",
    )


def _validate_mesh_geometry(mesh: cell_mesh_owner.CellMesh, /) -> None:
    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    for block in mesh.blocks:
        cells = np.asarray(block.vertices, dtype=np.int32)
        determinant, failure = _cell_metric_determinants(
            block.cell_kind, coordinates[cells]
        )
        if np.any(~np.isfinite(determinant)) or np.any(determinant <= 0.0):
            raise ValueError(failure)


def _validate_mixed_prism_tetrahedron_admission(
    mesh: cell_mesh_owner.CellMesh,
    resolved_fields: tuple[tuple[FiniteElementSpec, ...], ...],
    coordinate_spec: cell_geometry_owner.CellGeometrySpec,
    /,
    *,
    budget: CoordinateEnclosureBudget | None,
) -> None:
    if {block.cell_kind for block in mesh.blocks} != {"prism", "tetrahedron"}:
        return
    if not isinstance(mesh.connectivity, PolyhedralConnectivity):
        raise ValueError(
            "FiniteElementPlan.mesh.connectivity must be PolyhedralConnectivity for mixed prism/tetrahedron fields; got a different connectivity"
        )

    for elements in resolved_fields:
        for block, element in zip(mesh.blocks, elements, strict=True):
            nodal_h1 = (
                element.conformity == "H1"
                and element.mapping == "identity"
                and not element.value_shape
                and element.representation == "point_value"
            )
            local_dg = (
                element.mapping == "identity"
                and not element.value_shape
                and element.conformity == "L2"
                and element.element_id
                == discontinuous_element(block.cell_kind, element.degree).element_id
            )
            basis = element.form_basis
            canonical_form = False
            if type(basis) is form_elements_owner.FormBasis:
                dimension, family = form_elements_owner.admit_form_cell(
                    block.cell_kind, basis.family
                )
                tabulator = element.tabulator
                canonical_form = (
                    basis.dimension == dimension
                    and element.family == family
                    and basis.family == family
                    and element.degree == basis.order
                    and element.value_spec.form_type.degree == basis.form_degree
                    and element.value_spec.form_type.twist == basis.twist
                    and element.representation == "polynomial_moment"
                    and element.tabulator_id == basis.basis_id
                    and type(tabulator) is form_elements_owner.ProxyTabulator
                    and tabulator.value_spec == element.value_spec
                    and (
                        tabulator.basis is basis
                        or array_tree_fingerprint(tabulator.basis)
                        == array_tree_fingerprint(basis)
                    )
                )
            if not (nodal_h1 or local_dg or canonical_form):
                raise ValueError(
                    "FiniteElementPlan.fields must use form elements, nodal H1, or implemented "
                    "discontinuous scalar references on mixed prism/tetrahedron cells; got unsupported field semantics"
                )
    coordinate_elements, _, _ = coordinate_spec.resolve(mesh, budget=budget)
    if any(
        not isinstance(
            element,
            (
                FiniteElementSpec,
                cell_geometry_owner.BarycentricCellGeometryElement,
                cell_geometry_owner.RestrictedCellGeometryElement,
                cell_geometry_owner.PolynomialComposedCellGeometryElement,
                cell_geometry_owner.RationalComposedCellGeometryElement,
                cell_geometry_owner.SplineCellGeometryElement,
                cell_geometry_owner.LayerColumnCellGeometryElement,
            ),
        )
        for element in coordinate_elements
    ):
        raise TypeError(
            "FiniteElementPlan.coordinate_spec.elements must provide supported mixed-cell geometry tabulation; got an unsupported coordinate element"
        )


def _coefficient_dtype(value: DTypeLike, /) -> np.dtype:
    dtype = jnp.dtype(jax.dtypes.canonicalize_dtype(np.dtype(value)))
    if not jnp.issubdtype(dtype, jnp.inexact):
        raise TypeError(
            "FiniteElementPlan.coefficient_dtype must be real or complex floating point; got a non-inexact dtype"
        )
    return dtype


def _validate_vertex_usage(mesh: cell_mesh_owner.CellMesh, /) -> None:
    used_vertices = np.unique(
        np.concatenate(
            tuple(
                np.asarray(block.vertices, dtype=np.int32).reshape((-1,))
                for block in mesh.blocks
            )
        )
    )
    if used_vertices.size != mesh.coordinates.shape[0]:
        raise ValueError(
            "FiniteElementPlan.mesh.vertices must all occur in cells; got unused vertices"
        )


@final
class FiniteElementPlan(AbstractDiscretizationPlan):
    mesh: cell_mesh_owner.CellMesh
    fields: tuple[FiniteElementFieldSpec, ...]
    coordinate_spec: cell_geometry_owner.CellGeometrySpec
    precision_policy: FiniteElementPrecisionPolicy
    coefficient_dtype: np.dtype = eqx.field(static=True)
    coordinate_measure: CoordinateMeasurePlan | None
    key: DiscretizationKey
    capabilities: tuple[DiscretizationCapability, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        mesh: cell_mesh_owner.CellMesh,
        fields: FiniteElementFieldSpec | Sequence[FiniteElementFieldSpec],
        /,
        *,
        precision_policy: FiniteElementPrecisionPolicy | None = None,
        coordinate_spec: cell_geometry_owner.CellGeometrySpec | None = None,
        coefficient_dtype: DTypeLike = jnp.float64,
        coordinate_measure: CoordinateMeasurePlan | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> None:
        dtype = _coefficient_dtype(coefficient_dtype)
        _validate_mesh_geometry(mesh)
        _validate_vertex_usage(mesh)
        field_specs = (
            (fields,) if isinstance(fields, FiniteElementFieldSpec) else tuple(fields)
        )
        if not field_specs or not all(
            isinstance(field, FiniteElementFieldSpec) for field in field_specs
        ):
            raise TypeError(
                "FiniteElementPlan.fields must contain FiniteElementFieldSpec instances; got empty or incompatible fields"
            )
        names = tuple(field.name for field in field_specs)
        if len(set(names)) != len(names):
            raise ValueError(
                "FiniteElementPlan.fields.name must be unique; got duplicate field names"
            )
        resolved_fields = tuple(field.resolve(mesh) for field in field_specs)
        if coordinate_spec is None:
            coordinates = (
                cell_geometry_owner.CellGeometrySpec.affine(mesh)
                if mesh.storage is None
                else mesh.storage.restore_geometry()
            )
        else:
            coordinates = coordinate_spec
        _validate_mixed_prism_tetrahedron_admission(
            mesh,
            resolved_fields,
            coordinates,
            budget=budget,
        )
        if coordinate_measure is not None:
            _, _, coordinate_values = coordinates.resolve(mesh)
            coordinate_measure.validate_coordinates(coordinate_values)
            if coordinate_measure.kind == "axisymmetric-rz":
                if mesh.topological_dimension != 2 or mesh.ambient_dimension != 2:
                    raise ValueError(
                        "FiniteElementPlan.mesh.dimension must be two for axisymmetric RZ; got incompatible topological or ambient dimension"
                    )
            elif coordinate_measure.kind != "cartesian":
                if mesh.topological_dimension != 1 or mesh.ambient_dimension != 1:
                    raise ValueError(
                        "FiniteElementPlan.mesh.dimension must be one for radial reductions; got incompatible topological or ambient dimension"
                    )
        precision = (
            FiniteElementPrecisionPolicy()
            if precision_policy is None
            else precision_policy
        )
        self.mesh = mesh
        self.fields = field_specs
        self.coordinate_spec = coordinates
        self.precision_policy = precision
        self.coefficient_dtype = dtype
        self.coordinate_measure = coordinate_measure
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
                "coordinate_spec": coordinates.geometry_layout_id,
                "precision_policy": precision.policy_id,
                "coefficient_dtype": dtype.str,
                **(
                    {"coordinate_measure": coordinate_measure.plan_id}
                    if coordinate_measure is not None
                    and coordinate_measure.kind != "cartesian"
                    else {}
                ),
            }
        )

    @classmethod
    @checked
    def from_stored_fields(cls, value: FiniteElementPlan, /) -> FiniteElementPlan:
        """Reconstruct a plan within its coordinate preparation resource bounds."""
        work_upper, bytes_upper = cell_geometry_owner.cell_geometry_preparation_bounds(
            value.coordinate_spec
        )
        query_upper = cell_geometry_owner.cell_geometry_preparation_query_upper(
            value.coordinate_spec
        )
        with cell_geometry_owner.owned_coordinate_preparation(
            work_upper, bytes_upper, maximum_geometry_queries=query_upper
        ) as budget:
            return cls(
                value.mesh,
                value.fields,
                precision_policy=value.precision_policy,
                coordinate_spec=value.coordinate_spec,
                coefficient_dtype=value.coefficient_dtype,
                coordinate_measure=value.coordinate_measure,
                budget=budget,
            )

    def prepare(
        self,
        /,
        *,
        numeric_version: str = "0",
        reuse_prepared: FiniteElementDiscretization | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> FiniteElementDiscretization:
        if reuse_prepared is None:
            return FiniteElementDiscretization(
                self, numeric_version=numeric_version, budget=budget
            )
        reuse_prepared.require_exact_plan(
            self, numeric_version=numeric_version, budget=budget
        )
        return reuse_prepared

    def prepare_transfer(
        self,
        /,
        *,
        numeric_version: str = "0",
        reuse_prepared: FiniteElementTransferDiscretization | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> FiniteElementTransferDiscretization:
        """Prepare only immutable field layouts required by transfer operators."""
        if reuse_prepared is None:
            return FiniteElementTransferDiscretization(
                self,
                numeric_version=numeric_version,
                budget=budget,
            )
        reuse_prepared.require_exact_plan(
            self, numeric_version=numeric_version, budget=budget
        )
        return reuse_prepared


def _finite_element_field_layout(
    mesh: cell_mesh_owner.CellMesh,
    dof_map: FiniteElementDofMap,
    components: tuple[int, ...],
    /,
) -> EntityDofLayout | BlockDofLayout:
    if dof_map.coordinate_gather is None:
        entity_sets = mesh.topology.entity_sets
    else:
        periodic = mesh.periodic_topology
        if periodic is None:
            raise ValueError(
                "finite_element.dof_layout.mesh must provide periodic topology for quotient DOFs; got no periodic topology"
            )
        entity_sets = periodic.quotient.entity_sets
    if dof_map.association in ("vertex", "quotient_vertex"):
        return EntityDofLayout(
            entity_sets[0].entity_set_id,
            entity_sets[0].count,
            dof_map.global_dof_count,
            component_shape=components,
        )
    if dof_map.association in (
        "entity",
        "form_entity",
        "quotient_entity",
        "quotient_form_entity",
    ):
        entity_names = (
            ("vertices", "edges", "faces", "cells")
            if mesh.topological_dimension <= 3
            else tuple(
                f"entities_{degree}" for degree in range(mesh.topological_dimension + 1)
            )
        )
        block_names, layouts = [], []
        for dimension, (entity_dof_count, dofs_per_entity) in enumerate(
            zip(dof_map.entity_dof_counts, dof_map.entity_dofs_per_entity, strict=True)
        ):
            if entity_dof_count == 0:
                continue
            entities = entity_sets[dimension]
            block_names.append(entity_names[dimension])
            layouts.append(
                EntityDofLayout(
                    entities.entity_set_id,
                    entities.count,
                    entity_dof_count,
                    dofs_per_entity=dofs_per_entity,
                    component_shape=components,
                )
            )
        return BlockDofLayout(tuple(block_names), tuple(layouts))
    if dof_map.association == "edge":
        edge_count = dof_map.global_dof_count
        return EntityDofLayout(
            mesh.topology.entity_sets[1].entity_set_id,
            edge_count,
            edge_count,
            component_shape=components,
        )
    if dof_map.association == "cell":
        cell_dof_count = dof_map.global_dof_count
        return EntityDofLayout(
            mesh.topology.entity_sets[mesh.topological_dimension].entity_set_id,
            cell_dof_count,
            cell_dof_count,
            component_shape=components,
        )
    raise ValueError(
        "finite_element.dof_layout.association must select a supported DOF association; got an unsupported association"
    )


def finite_element_field_space(
    mesh: cell_mesh_owner.CellMesh,
    field: FiniteElementFieldSpec,
    elements: tuple[FiniteElementSpec, ...],
    dof_map: FiniteElementDofMap,
    coefficient_dtype: np.dtype,
    /,
) -> DiscreteFieldSpace:
    """Bind a field's discrete space to its resolved elements and DOF map."""
    vector_shape = (dof_map.global_dof_count,) + field.component_shape
    vector_space = ArraySpace(vector_shape, dtype=coefficient_dtype)
    layout = _finite_element_field_layout(mesh, dof_map, field.component_shape)
    conformity = elements[0].conformity
    representation = (
        "rational_moment"
        if any(element.representation == "rational_moment" for element in elements)
        else elements[0].representation
    )
    return DiscreteFieldSpace(
        field.name,
        mesh.support.support_id,
        layout,
        vector_space,
        representation=representation,
        conformity=conformity,
        form_type=FormType(
            elements[0].value_spec.form_type.dimension,
            elements[0].value_spec.form_type.degree,
            twist=elements[0].value_spec.form_type.twist,
            fiber_shape=elements[0].value_spec.form_type.fiber_shape
            + field.component_shape,
            ambient_dimension=mesh.ambient_dimension,
        ),
        projection_id=canonical_fingerprint(
            {"kind": "finite-element-projection", "field": field.field_spec_id}
        ),
        reconstruction_id=canonical_fingerprint(
            {"kind": "finite-element-reconstruction", "field": field.field_spec_id}
        ),
    )


def _prepare_finite_element_field_layout(
    mesh: cell_mesh_owner.CellMesh,
    field: FiniteElementFieldSpec,
    coefficient_dtype: np.dtype,
    coordinate_spec: cell_geometry_owner.CellGeometrySpec,
    /,
    *,
    dof_map: FiniteElementDofMap | None = None,
    budget: CoordinateEnclosureBudget | None,
    coordinate_preparation: _PreparedCoordinateMaps | None = None,
) -> tuple[
    tuple[FiniteElementSpec, ...],
    FiniteElementDofMap,
    DiscreteFieldSpace,
]:
    elements = field.resolve(mesh)
    match dof_map:
        case None:
            dof_map = FiniteElementDofMap(
                mesh,
                elements,
                component_shape=field.component_shape,
                coordinate_spec=coordinate_spec,
                budget=budget,
                _coordinate_preparation=coordinate_preparation,
            )
        case FiniteElementDofMap():
            if (
                dof_map.block_names != tuple(block.name for block in mesh.blocks)
                or dof_map.element_ids
                != tuple(element.element_id for element in elements)
                or dof_map.component_shape != field.component_shape
                or any(
                    rows.shape != (block.cell_count, element.local_dof_count)
                    for rows, block, element in zip(
                        dof_map.cell_dofs, mesh.blocks, elements, strict=True
                    )
                )
            ):
                raise ValueError(
                    "FiniteElementDofMap.projection must match resident cells and field references; "
                    "got incompatible metadata"
                )
        case _:
            raise ValueError(
                "FiniteElementDofMap.projection must match resident cells and field references; "
                "got incompatible metadata"
            )
    return (
        elements,
        dof_map,
        finite_element_field_space(mesh, field, elements, dof_map, coefficient_dtype),
    )


def _prepare_finite_element_field(
    mesh: cell_mesh_owner.CellMesh,
    field: FiniteElementFieldSpec,
    coefficient_dtype: np.dtype,
    coordinate_elements: tuple[CellGeometryElement, ...],
    coordinate_dofs: tuple[Array, ...],
    coordinate_values: Array,
    precision_policy: FiniteElementPrecisionPolicy,
    coordinate_spec: cell_geometry_owner.CellGeometrySpec,
    /,
    *,
    dof_map: FiniteElementDofMap | None = None,
    coordinate_measure: CoordinateMeasurePlan | None = None,
    budget: CoordinateEnclosureBudget | None,
    coordinate_preparation: _PreparedCoordinateMaps | None = None,
) -> tuple[
    tuple[FiniteElementSpec, ...],
    FiniteElementDofMap,
    DiscreteFieldSpace,
    tuple[FiniteElementBlockGeometry, ...],
]:
    elements, dof_map, space = _prepare_finite_element_field_layout(
        mesh,
        field,
        coefficient_dtype,
        coordinate_spec,
        dof_map=dof_map,
        budget=budget,
        coordinate_preparation=coordinate_preparation,
    )
    geometries = tuple(
        _prepare_block_geometry(
            mesh,
            block,
            element,
            coordinates=coordinate_values,
            coordinate_element=coordinate_element,
            geometry_dofs=geometry_dofs,
            precision_policy=precision_policy,
            coordinate_measure=coordinate_measure,
            _preparation_bucket=True,
        )
        for block, element, coordinate_element, geometry_dofs in zip(
            mesh.blocks, elements, coordinate_elements, coordinate_dofs, strict=True
        )
    )
    return elements, dof_map, space, geometries


def _finite_element_domains(
    mesh: cell_mesh_owner.CellMesh, routes: tuple[np.ndarray, ...], /
) -> tuple[IntegrationDomain, IntegrationDomain, IntegrationDomain]:
    owner, neighbor, owner_local, neighbor_local = routes
    exterior = np.flatnonzero(neighbor < 0)
    interior = np.flatnonzero(neighbor >= 0)
    cell_count = sum(block.cell_count for block in mesh.blocks)
    cell_entities = mesh.topology.entity_sets[mesh.topological_dimension]
    facet_entities = mesh.topology.entity_sets[mesh.topological_dimension - 1]
    cell_domain = IntegrationDomain(
        "cell",
        np.arange(cell_count, dtype=np.int32),
        mesh.support.support_id,
        cell_entities.entity_set_id,
        owner_cells=np.arange(cell_count, dtype=np.int32),
    )
    exterior_domain = IntegrationDomain(
        "exterior_facet",
        exterior,
        mesh.support.support_id,
        facet_entities.entity_set_id,
        owner_cells=owner[exterior],
        neighbor_cells=neighbor[exterior],
        owner_local_entities=owner_local[exterior],
        neighbor_local_entities=neighbor_local[exterior],
    )
    interior_domain = IntegrationDomain(
        "interior_facet",
        interior,
        mesh.support.support_id,
        facet_entities.entity_set_id,
        owner_cells=owner[interior],
        neighbor_cells=neighbor[interior],
        owner_local_entities=owner_local[interior],
        neighbor_local_entities=neighbor_local[interior],
    )
    return cell_domain, exterior_domain, interior_domain


@eqx.filter_jit
def _project_form_block(
    cell_map: PreparedFiniteElementCellMap,
    element: FiniteElementSpec,
    coordinates: Array,
    transforms: Array,
    function: Callable[[Array, object], ArrayLike],
    args: object,
    /,
) -> Array:
    """Fuse independent moment evaluations while retaining each callback's axes."""
    basis = element.form_basis
    if basis is None:
        raise ValueError(
            "FiniteElementSpec.form_basis must describe projection moments; got None"
        )
    value_spec = element.value_spec
    form = value_spec.form_type
    embedded = cell_map.ambient_dimension != cell_map.reference_dimension
    physical_spec = (
        FormValueSpec(
            FormType(
                form.dimension,
                form.degree,
                twist=form.twist,
                fiber_shape=form.fiber_shape,
                ambient_dimension=cell_map.ambient_dimension,
            ),
            proxy=value_spec.proxy,
        )
        if embedded
        else value_spec
    )

    def project_cell(cell: Array, transform: Array, /) -> Array:
        cells = jnp.full((basis.functional_points.shape[0],), cell, dtype=jnp.int32)
        geometry = cell_map.evaluate(coordinates, cells, basis.functional_points)
        physical = jnp.asarray(function(geometry.physical_points, args))
        if embedded and value_spec.proxy == "flux":
            intrinsic = ein.contract("qrd,qd->qr", geometry.inverse_jacobian, physical)
            reference = vector_to_form(intrinsic * geometry.measure[:, None], value_spec)
        elif embedded and value_spec.proxy == "density":
            reference = vector_to_form(physical * geometry.measure, value_spec)
        else:
            values = vector_to_form(physical, physical_spec)
            reference = pullback(values, physical_spec.form_type, geometry.jacobian)
        return jnp.linalg.solve(transform, basis.interpolate(reference))

    return jax.vmap(project_cell)(
        jnp.arange(cell_map.cell_count, dtype=jnp.int32), transforms
    )


def _require_exact_finite_element_plan(
    prepared: FiniteElementTransferDiscretization | FiniteElementDiscretization,
    plan: FiniteElementPlan,
    numeric_version: str,
    budget: CoordinateEnclosureBudget | None,
    errors: tuple[str, str, str],
    /,
) -> None:
    """Validate reuse against ordered declarations and bitwise coordinate state."""
    version = str(numeric_version)
    if (
        not version
        or prepared.numeric_version != version
        or prepared.mesh.topology_id != plan.mesh.topology_id
        or tuple(field.field_spec_id for field in prepared.construction_plan.fields)
        != tuple(field.field_spec_id for field in plan.fields)
        or tuple(space.name for space in prepared.field_spaces)
        != tuple(field.name for field in plan.fields)
        or prepared.precision_policy.policy_id != plan.precision_policy.policy_id
        or prepared.coefficient_dtype != plan.coefficient_dtype
    ):
        raise ValueError(errors[0])
    elements, routes, values = plan.coordinate_spec.resolve(plan.mesh, budget=budget)
    if (
        tuple(element.element_id for element in elements)
        != tuple(element.element_id for element in prepared.coordinate_elements)
        or any(
            not np.array_equal(np.asarray(actual), np.asarray(retained))
            for actual, retained in zip(routes, prepared.coordinate_dofs, strict=True)
        )
        or not np.array_equal(
            np.asarray(values).view(np.uint64),
            np.asarray(prepared.default_runtime.coordinates).view(np.uint64),
        )
    ):
        raise ValueError(errors[1])
    resolved = tuple(field.resolve(plan.mesh) for field in plan.fields)
    for field, expected, retained_elements, dof_map in zip(
        plan.fields, resolved, prepared.elements, prepared.dof_maps, strict=True
    ):
        element_ids = tuple(element.element_id for element in expected)
        if (
            element_ids != tuple(element.element_id for element in retained_elements)
            or element_ids != dof_map.element_ids
            or dof_map.block_names != tuple(block.name for block in plan.mesh.blocks)
            or dof_map.component_shape != field.component_shape
            or dof_map.construction_mesh.topology_id != plan.mesh.topology_id
        ):
            raise ValueError(errors[2])


@final
class FiniteElementTransferDiscretization(StrictModule, NonTrainableState):
    """Immutable FE basis/DOF preparation without variational geometry kernels.

    Nested transfer preparation needs basis, orientation, coordinate,
    and field-space metadata, but never assembled physical quadrature tables.
    This owner keeps that smaller contract explicit rather than compiling unused
    block assembly kernels in a cold transfer-only consumer.
    """

    mesh: cell_mesh_owner.CellMesh
    dof_maps: tuple[FiniteElementDofMap, ...]
    default_runtime: FiniteElementRuntimeData
    elements: tuple[tuple[FiniteElementSpec, ...], ...]
    coordinate_elements: tuple[CellGeometryElement, ...]
    coordinate_dofs: tuple[Array, ...]
    key: DiscretizationKey
    support: DiscreteSupport
    field_spaces: tuple[DiscreteFieldSpace, ...]
    precision_policy: FiniteElementPrecisionPolicy
    coefficient_dtype: np.dtype = eqx.field(static=True)
    capabilities: tuple[DiscretizationCapability, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    numeric_version: str = eqx.field(static=True)
    coordinate_spec: cell_geometry_owner.CellGeometrySpec
    construction_plan: FiniteElementPlan
    construction_projected_maps: tuple[FiniteElementDofMap, ...] | None

    @checked
    def __init__(
        self,
        plan: FiniteElementPlan,
        /,
        *,
        numeric_version: str = "0",
        dof_maps: Sequence[FiniteElementDofMap] | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> None:
        version = str(numeric_version)
        if not version:
            raise ValueError(
                "finite_element.numeric_version must be nonempty; got an empty revision"
            )
        mesh = plan.mesh
        coordinate_preparation = _PreparedCoordinateMaps.prepare(
            mesh, plan.coordinate_spec, budget=budget
        )
        coordinate_elements, coordinate_dofs, coordinate_values = (
            coordinate_preparation.elements,
            coordinate_preparation.routes,
            coordinate_preparation.coordinates,
        )
        projected_maps = None if dof_maps is None else tuple(dof_maps)
        if projected_maps is not None and len(projected_maps) != len(plan.fields):
            raise ValueError(
                "finite_element.dof_maps must contain one projected DOF map per field; got a different map count"
            )
        prepared_layouts: dict[
            tuple[tuple[str, ...], tuple[int, ...], str | None],
            tuple[tuple[FiniteElementSpec, ...], FiniteElementDofMap],
        ] = {}
        all_elements: list[tuple[FiniteElementSpec, ...]] = []
        all_dof_maps: list[FiniteElementDofMap] = []
        field_spaces: list[DiscreteFieldSpace] = []
        for field_index, field in enumerate(plan.fields):
            resolved = field.resolve(mesh)
            projected = None if projected_maps is None else projected_maps[field_index]
            layout_key = (
                tuple(element.element_id for element in resolved),
                field.component_shape,
                None if projected is None else projected.dof_map_id,
            )
            retained = prepared_layouts.get(layout_key)
            if retained is None:
                elements, dof_map, field_space = _prepare_finite_element_field_layout(
                    mesh,
                    field,
                    plan.coefficient_dtype,
                    plan.coordinate_spec,
                    dof_map=projected,
                    budget=budget,
                    coordinate_preparation=coordinate_preparation,
                )
                prepared_layouts[layout_key] = elements, dof_map
            else:
                elements, dof_map = retained
                if tuple(element.element_id for element in elements) != tuple(
                    element.element_id for element in resolved
                ) or (
                    projected is not None and projected.dof_map_id != dof_map.dof_map_id
                ):
                    raise ValueError(
                        "FiniteElementTransferDiscretization.shared_layout must preserve basis "
                        "and projected DOF orientation; got incompatible shared preparation"
                    )
                field_space = finite_element_field_space(
                    mesh,
                    field,
                    elements,
                    dof_map,
                    plan.coefficient_dtype,
                )
            all_elements.append(elements)
            all_dof_maps.append(dof_map)
            field_spaces.append(field_space)
        self.mesh = mesh
        self.dof_maps = tuple(all_dof_maps)
        self.default_runtime = FiniteElementRuntimeData(
            mesh,
            coordinate_values,
            numeric_version=version,
            geometry_layout_id=plan.coordinate_spec.geometry_layout_id,
        )
        self.elements = tuple(all_elements)
        self.coordinate_elements = coordinate_elements
        self.coordinate_dofs = coordinate_dofs
        self.key = plan.key
        self.support = mesh.support
        self.field_spaces = tuple(field_spaces)
        self.precision_policy = plan.precision_policy
        self.coefficient_dtype = plan.coefficient_dtype
        self.capabilities = plan.capabilities
        self.plan_id = plan.plan_id
        self.numeric_version = version
        self.coordinate_spec = plan.coordinate_spec
        self.construction_plan = plan
        self.construction_projected_maps = projected_maps
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-finite-element-transfer-space",
                "plan": plan.plan_id,
                "mesh": mesh.mesh_id,
                "numeric_version": version,
                "dof_maps": [value.dof_map_id for value in all_dof_maps],
            }
        )

    def field_index(self, field_name: str, /) -> int:
        requested = str(field_name)
        for index, field_space in enumerate(self.field_spaces):
            if field_space.name == requested:
                return index
        raise KeyError(
            f"FiniteElementTransferDiscretization.field_name must name a prepared field; got {requested!r}"
        )

    @checked
    def require_exact_plan(
        self,
        plan: FiniteElementPlan,
        /,
        *,
        numeric_version: str = "0",
        budget: CoordinateEnclosureBudget | None = None,
    ) -> None:
        """Require reusable mesh, map, basis, orientation, and role identity."""
        _require_exact_finite_element_plan(
            self,
            plan,
            numeric_version,
            budget,
            (
                "FiniteElementTransferDiscretization.plan must preserve revision, "
                "ordered fields, precision, and coefficient space; got incompatible metadata",
                "FiniteElementTransferDiscretization.coordinate_map must preserve "
                "basis, routes, and coefficient bits; got a different map",
                "FiniteElementTransferDiscretization.field_layout must preserve "
                "basis, orientation, roles, and source topology; got a different layout",
            ),
        )


@final
class FiniteElementDiscretization(AbstractPreparedLocalDiscretization):
    mesh: cell_mesh_owner.CellMesh
    dof_maps: tuple[FiniteElementDofMap, ...]
    default_runtime: FiniteElementRuntimeData
    elements: tuple[tuple[FiniteElementSpec, ...], ...]
    coordinate_elements: tuple[CellGeometryElement, ...]
    coordinate_dofs: tuple[Array, ...]
    block_geometries: tuple[tuple[FiniteElementBlockGeometry, ...], ...]
    cell_domain: IntegrationDomain
    exterior_facet_domain: IntegrationDomain
    interior_facet_domain: IntegrationDomain
    key: DiscretizationKey
    support: DiscreteSupport
    field_spaces: tuple[DiscreteFieldSpace, ...]
    block_space: BlockSpace
    measures: tuple[DiscreteMeasure, ...]
    precision_policy: FiniteElementPrecisionPolicy
    coefficient_dtype: np.dtype = eqx.field(static=True)
    coordinate_measure: CoordinateMeasurePlan | None
    capabilities: tuple[DiscretizationCapability, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    numeric_version: str = eqx.field(static=True)
    preparation: PreparationReport
    coordinate_spec: cell_geometry_owner.CellGeometrySpec
    construction_plan: FiniteElementPlan
    construction_projected_maps: tuple[FiniteElementDofMap, ...] | None

    @checked
    def __init__(
        self,
        plan: FiniteElementPlan,
        /,
        *,
        numeric_version: str = "0",
        dof_maps: Sequence[FiniteElementDofMap] | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> None:
        version = str(numeric_version)
        if not version:
            raise ValueError(
                "finite_element.numeric_version must be nonempty; got an empty revision"
            )
        mesh = plan.mesh
        coordinate_preparation = _PreparedCoordinateMaps.prepare(
            mesh, plan.coordinate_spec, budget=budget
        )
        coordinate_elements, coordinate_dofs, coordinate_values = (
            coordinate_preparation.elements,
            coordinate_preparation.routes,
            coordinate_preparation.coordinates,
        )
        projected_maps = None if dof_maps is None else tuple(dof_maps)
        if projected_maps is not None and len(projected_maps) != len(plan.fields):
            raise ValueError(
                "finite_element.dof_maps must contain one projected DOF map per field; got a different map count"
            )
        field_spaces = []
        dof_maps = []
        all_elements = []
        all_geometries = []
        shared_fields: dict[
            tuple[tuple[str, ...], tuple[int, ...], str | None],
            tuple[
                tuple[FiniteElementSpec, ...],
                FiniteElementDofMap,
                tuple[FiniteElementBlockGeometry, ...],
            ],
        ] = {}
        cell_measures = None
        for field_index, field in enumerate(plan.fields):
            resolved = field.resolve(mesh)
            projected = None if projected_maps is None else projected_maps[field_index]
            key = (
                tuple(element.element_id for element in resolved),
                field.component_shape,
                None if projected is None else projected.dof_map_id,
            )
            shared = shared_fields.get(key)
            if shared is None:
                elements, dof_map, field_space, geometries = (
                    _prepare_finite_element_field(
                        mesh,
                        field,
                        plan.coefficient_dtype,
                        coordinate_elements,
                        coordinate_dofs,
                        coordinate_values,
                        plan.precision_policy,
                        plan.coordinate_spec,
                        dof_map=projected,
                        coordinate_measure=plan.coordinate_measure,
                        budget=budget,
                        coordinate_preparation=coordinate_preparation,
                    )
                )
                shared_fields[key] = elements, dof_map, geometries
            else:
                elements, dof_map, geometries = shared
                if tuple(element.element_id for element in elements) != tuple(
                    element.element_id for element in resolved
                ) or (
                    projected is not None and projected.dof_map_id != dof_map.dof_map_id
                ):
                    raise ValueError(
                        "FiniteElementDiscretization.shared_layout must preserve basis and "
                        "projected DOF orientation; got incompatible shared preparation"
                    )
                field_space = finite_element_field_space(
                    mesh,
                    field,
                    elements,
                    dof_map,
                    plan.coefficient_dtype,
                )
            all_elements.append(elements)
            dof_maps.append(dof_map)
            field_spaces.append(field_space)
            if (
                plan.coordinate_measure is not None
                and plan.coordinate_measure.kind != "cartesian"
            ):
                for geometry in geometries:
                    plan.coordinate_measure.validate_coordinates(geometry.physical_points)
                    if np.any(np.asarray(geometry.physical_weights) <= 0.0):
                        raise ValueError(
                            "FiniteElementDiscretization.rotational_quadrature must lie strictly off the axis; got nonpositive physical weights"
                        )
            all_geometries.append(geometries)
            if cell_measures is None:
                cell_measures = jnp.concatenate(
                    tuple(geometry.measure for geometry in geometries),
                    axis=0,
                )
        if cell_measures is None:
            raise ValueError(
                "FiniteElementDiscretization.cell_measures must contain prepared measures; got no cell measures"
            )
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
                "vertices": mesh.coordinates.shape[0],
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
        facet_routes = _facet_routes(mesh)
        self.mesh = mesh
        self.coordinate_spec = plan.coordinate_spec
        self.default_runtime = FiniteElementRuntimeData(
            mesh,
            coordinate_values,
            numeric_version=version,
            geometry_layout_id=plan.coordinate_spec.geometry_layout_id,
            coordinate_frame=(
                plan.coordinate_measure.frame
                if plan.coordinate_measure is not None
                else "reference"
            ),
        )
        self.dof_maps = tuple(dof_maps)
        self.elements = tuple(all_elements)
        self.coordinate_elements = coordinate_elements
        self.coordinate_dofs = coordinate_dofs
        self.block_geometries = tuple(all_geometries)
        self.cell_domain, self.exterior_facet_domain, self.interior_facet_domain = (
            _finite_element_domains(mesh, facet_routes)
        )
        self.key = plan.key
        self.precision_policy = plan.precision_policy
        self.coefficient_dtype = plan.coefficient_dtype
        self.coordinate_measure = plan.coordinate_measure
        self.support = mesh.support
        self.block_space = BlockSpace(
            tuple(space.vector_space for space in spaces),
            names=tuple(space.name for space in spaces),
        )
        self.field_spaces = spaces
        self.measures = measures
        self.capabilities = capabilities
        self.plan_id = plan.plan_id
        self.numeric_version = version
        self.preparation = preparation
        self.construction_plan = plan
        self.construction_projected_maps = projected_maps
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-finite-element",
                "plan": plan.plan_id,
                "mesh": mesh.mesh_id,
                "numeric_version": version,
            }
        )

    @checked
    def require_exact_plan(
        self,
        plan: FiniteElementPlan,
        /,
        *,
        numeric_version: str = "0",
        budget: CoordinateEnclosureBudget | None = None,
    ) -> None:
        """Require that this immutable preparation realizes ``plan`` bitwise.

        This admits reuse only for an unchanged mesh revision, coordinate map,
        ordered field roles, reference bases, and shared DOF orientations.
        Geometry-transition provenance may differ outside the coordinate map;
        every resolved element, route, and coefficient must still be bitwise
        identical.
        """
        _require_exact_finite_element_plan(
            self,
            plan,
            numeric_version,
            budget,
            (
                "FiniteElementDiscretization.plan must preserve revision, ordered fields, "
                "precision, and coefficient space; got incompatible metadata",
                "FiniteElementDiscretization.coordinate_map must preserve basis, "
                "routes, and coefficient bits; got a different map",
                "FiniteElementDiscretization.field_layout must preserve basis, "
                "orientation, roles, and source topology; got a different layout",
            ),
        )

    @authenticate_restored_node
    def validate_restored(
        self,
        /,
        *,
        validation_context: RestorationValidationContext,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> None:
        """Replay the consumer, including geometry and field caches."""
        for dof_map in self.dof_maps:
            dof_map.validate_restored(validation_context=validation_context)
        rebuilt = FiniteElementDiscretization(
            self.construction_plan,
            numeric_version=self.numeric_version,
            dof_maps=self.construction_projected_maps,
            budget=budget,
        )
        validate_restored_fem_state(self, rebuilt)

    @property
    def precision_evidence(self) -> Any:
        return self.precision_policy.evidence()

    def prepare_runtime(
        self,
        coordinates: ArrayLike | None = None,
        /,
        *,
        numeric_version: str,
        coordinate_frame: CoordinateFrame | None = None,
    ) -> FiniteElementRuntimeData:
        points = jnp.asarray(
            self.default_runtime.coordinates if coordinates is None else coordinates
        )
        if points.shape != self.default_runtime.coordinates.shape:
            raise ValueError(
                "FiniteElementDiscretization.runtime.coordinates must preserve coordinate shape; got incompatible shape"
            )
        frame = (
            self.default_runtime.coordinate_frame
            if coordinate_frame is None
            else coordinate_frame
        )
        if self.coordinate_measure is not None and frame != self.coordinate_measure.frame:
            raise ValueError(
                "FiniteElementDiscretization.runtime.coordinate_frame must match the coordinate measure; got a different frame"
            )
        return FiniteElementRuntimeData(
            self.mesh,
            points,
            numeric_version=numeric_version,
            geometry_layout_id=self.default_runtime.geometry_layout_id,
            coordinate_frame=frame,
        )

    def local_variational_capabilities(self, /) -> LocalVariationalCapabilities:

        return local_provider_owner.FiniteElementLocalProvider(
            self
        ).local_variational_capabilities()

    def local_field_binding(self, name: str, /) -> LocalFieldBinding:

        return local_provider_owner.FiniteElementLocalProvider(self).local_field_binding(
            name
        )

    def prepare_local_regions(
        self,
        domain: IntegrationDomain,
        /,
        *,
        field_names: tuple[str, ...],
        maximum_derivative_order: int,
        kernel_mode: str,
    ) -> tuple[PreparedLocalRegion, ...]:

        return local_provider_owner.FiniteElementLocalProvider(
            self
        ).prepare_local_regions(
            domain,
            field_names=field_names,
            maximum_derivative_order=maximum_derivative_order,
            kernel_mode=kernel_mode,
        )

    def validate_local_runtime(self, runtime: object, /) -> None:

        local_provider_owner.FiniteElementLocalProvider(self).validate_local_runtime(
            runtime
        )

    @property
    def mass(self) -> SparseLinearMap:
        if len(self.field_spaces) != 1:
            raise ValueError(
                "FiniteElementDiscretization.mass.fields must contain one field; got multiple fields"
            )
        dof_map, geometries = self._field_operator_geometry(
            self.field_spaces[0].name, self.default_runtime
        )
        return self._assemble_geometry_operator(dof_map, geometries, mass=True)

    @property
    def stiffness(self) -> SparseLinearMap:
        if len(self.field_spaces) != 1:
            raise ValueError(
                "FiniteElementDiscretization.stiffness.fields must contain one field; got multiple fields"
            )
        dof_map, geometries = self._field_operator_geometry(
            self.field_spaces[0].name, self.default_runtime
        )
        return self._assemble_geometry_operator(dof_map, geometries, mass=False)

    @property
    def vertices(self) -> Array:
        return self.mesh.coordinates

    @property
    def boundary_dof_mask(self) -> Array:
        if len(self.dof_maps) != 1:
            raise ValueError(
                "FiniteElementDiscretization.boundary_dof_mask.fields must contain one field; got multiple fields"
            )
        return self.dof_maps[0].boundary_dof_mask

    def axis_dof_mask(self, field_name: str, /) -> Array:
        """Identify regular-axis DOFs for the owner's radial displacement condition.

        Radial displacement must vanish here, unlike temperature or axial motion.
        """
        measure = self.coordinate_measure
        if measure is None or measure.radial_axis is None:
            raise ValueError(
                "FiniteElementDiscretization.coordinate_measure must declare a rotational radial axis; got no rotational measure"
            )
        dof_map = self.dof_maps[self.field_index(field_name)]
        return dof_map.dof_coordinates[..., measure.radial_axis] == 0.0

    @checked
    def dof_indices(
        self,
        field_name: str,
        selection: EntitySelection,
        /,
    ) -> Array:
        """Return sorted P1 DOFs incident to selected cells or facets."""

        field_index = self.field_index(field_name)
        dof_map = self.dof_maps[field_index]
        field_elements = self.elements[field_index]
        if dof_map.association != "vertex" or any(
            element.conformity != "H1" or element.degree != 1
            for element in field_elements
        ):
            raise ValueError(
                "FiniteElementDiscretization.entity_selection.field must be vertex-associated P1 H1; got unsupported field semantics"
            )
        selected_entities = tuple(
            entities
            for entities in self.mesh.topology.entity_sets
            if entities.entity_set_id == selection.entity_set_id
        )
        if len(selected_entities) != 1:
            raise ValueError(
                "FiniteElementDiscretization.entity_selection.entity_set_id must belong to the mesh topology; got a different entity set"
            )
        entities = selected_entities[0]
        dimension = entities.intrinsic_dimension
        if dimension not in (
            self.mesh.topological_dimension,
            self.mesh.topological_dimension - 1,
        ):
            raise ValueError(
                "FiniteElementDiscretization.entity_selection.dimension must identify mesh cells or facets; got unsupported entity dimension"
            )
        entity_mask = np.asarray(selection.mask, dtype=np.bool_)
        active_mask = np.asarray(selection.active_mask, dtype=np.bool_)
        expected_active = np.asarray(entities.active_mask, dtype=np.bool_)
        if (
            entity_mask.shape != (entities.count,)
            or active_mask.shape != (entities.count,)
            or not np.array_equal(active_mask, expected_active)
            or np.any(entity_mask & ~expected_active)
        ):
            raise ValueError(
                "FiniteElementDiscretization.entity_selection.mask must match entity capacity and active mask; got incompatible selection masks"
            )
        for degree in range(dimension, 0, -1):
            incidence = self.mesh.topology.incidences[degree - 1]
            relation = incidence.relation
            valid = np.asarray(relation.valid, dtype=np.bool_)
            source = np.asarray(relation.source_indices, dtype=np.int32)
            target = np.asarray(relation.target_indices, dtype=np.int32)
            routes = valid & entity_mask[target]
            lower_entities = self.mesh.topology.entity_sets[degree - 1]
            lower_mask = np.zeros((lower_entities.count,), dtype=np.bool_)
            lower_mask[source[routes]] = True
            entity_mask = lower_mask & np.asarray(
                lower_entities.active_mask,
                dtype=np.bool_,
            )
        indices = np.flatnonzero(entity_mask).astype(np.int32)
        if indices.size and int(indices[-1]) >= dof_map.global_dof_count:
            raise ValueError(
                "FiniteElementDiscretization.entity_selection.dofs must lie in the field DOF map; got incompatible topology or DOF indices"
            )
        return jnp.asarray(indices, dtype=jnp.int32)

    def dof_mask(
        self,
        field_name: str,
        selection: EntitySelection,
        /,
    ) -> Array:
        """Project a supported cell/facet selection onto the P1 DOF axis."""

        field_index = self.field_index(field_name)
        mask = np.zeros((self.dof_maps[field_index].global_dof_count,), dtype=np.bool_)
        mask[np.asarray(self.dof_indices(field_name, selection), dtype=np.int32)] = True
        return jnp.asarray(mask)

    def project(
        self,
        field_name: str,
        function: Callable[[Array, object], ArrayLike],
        /,
        *,
        runtime: FiniteElementRuntimeData | None = None,
        args: object = None,
    ) -> Array:
        if not callable(function):
            raise TypeError(
                "FiniteElementDiscretization.function must be callable; got a noncallable object"
            )
        field_index = self.field_index(field_name)
        realized = self.default_runtime if runtime is None else runtime
        if self.elements[field_index][0].form_basis is not None:
            # Lazy: fem._generic <-> _cell_map import cycle.
            from . import _cell_map as cell_map_owner

            dofs = self.dof_maps[field_index]
            moments = jnp.zeros((dofs.global_dof_count,), dtype=self.coefficient_dtype)
            counts = jnp.zeros((dofs.global_dof_count,), dtype=jnp.int32)
            for block_index, element in enumerate(self.elements[field_index]):
                cell_map = cell_map_owner.PreparedFiniteElementCellMap(self, block_index)
                local = _project_form_block(
                    cell_map,
                    element,
                    realized.coordinates,
                    dofs.cell_transforms[block_index],
                    function,
                    args,
                )
                routes = dofs.cell_dofs[block_index]
                moments = moments.at[routes].add(local)
                counts = counts.at[routes].add(1)
            return moments / counts
        coordinates = self.dof_maps[field_index].evaluate_coordinates(
            self.mesh,
            realized.coordinates,
        )
        values = jnp.asarray(function(coordinates, args))
        return self.field_spaces[field_index].vector_space.validate(values)

    @checked
    def integration_domain(
        self,
        kind: str,
        selection: EntitySelection | None = None,
        /,
    ) -> IntegrationDomain:
        domains = {
            "cell": self.cell_domain,
            "exterior_facet": self.exterior_facet_domain,
            "interior_facet": self.interior_facet_domain,
        }
        if kind not in domains:
            raise ValueError(
                "FiniteElementDiscretization.domain.kind must select a supported integration domain; got an unsupported kind"
            )
        base = domains[kind]
        if selection is None:
            return base
        if selection.entity_set_id != base.entity_set_id:
            raise ValueError(
                "FiniteElementDiscretization.domain.selection.entity_set_id must match the domain entity set; got a different entity set"
            )
        entity_mask = np.asarray(selection.mask, dtype=np.bool_)
        base_entities = np.asarray(base.entity_indices, dtype=np.int32)
        selected_rows = np.flatnonzero(entity_mask[base_entities])
        return IntegrationDomain(
            base.kind,
            base_entities[selected_rows],
            base.support_id,
            base.entity_set_id,
            owner_cells=np.asarray(base.owner_cells)[selected_rows],
            neighbor_cells=np.asarray(base.neighbor_cells)[selected_rows],
            owner_local_entities=np.asarray(base.owner_local_entities)[selected_rows],
            neighbor_local_entities=np.asarray(base.neighbor_local_entities)[
                selected_rows
            ],
            neighbor_trace_permutations=np.asarray(base.neighbor_trace_permutations)[
                selected_rows
            ],
            periodic_face_mask=np.asarray(base.periodic_face_mask)[selected_rows],
            selection_id=selection.selection_id,
        )

    def cell_block_domain(self, block_name: str, /) -> IntegrationDomain:
        """Return the global cell domain for one named mesh block."""

        name = str(block_name)
        names = tuple(block.name for block in self.mesh.blocks)
        if name not in names:
            raise ValueError(
                f"FiniteElementDiscretization.block_name must name a mesh block; got {name!r}"
            )
        index = names.index(name)
        start = sum(block.cell_count for block in self.mesh.blocks[:index])
        stop = start + self.mesh.blocks[index].cell_count
        entities = np.arange(start, stop, dtype=np.int32)
        return IntegrationDomain(
            "cell",
            entities,
            self.cell_domain.support_id,
            self.cell_domain.entity_set_id,
            owner_cells=entities,
            selection_id=canonical_fingerprint(
                {
                    "kind": "finite-element-cell-block-domain",
                    "prepared": self.prepared_id,
                    "block": name,
                }
            ),
        )

    def trace(
        self,
        field_name: str,
        coefficients: ArrayLike,
        /,
        *,
        runtime: FiniteElementRuntimeData | None = None,
    ) -> tuple[Array, Array]:
        field_index = self.field_index(field_name)
        values = self.field_spaces[field_index].vector_space.validate(coefficients)
        realized = self.default_runtime if runtime is None else runtime
        coordinates = self.dof_maps[field_index].evaluate_coordinates(
            self.mesh,
            realized.coordinates,
        )
        mask = self.dof_maps[field_index].boundary_dof_mask
        return coordinates[mask], values[mask]

    def prepare_side_trace(
        self,
        field_name: str,
        domain: IntegrationDomain,
        /,
        *,
        rule: FacetTraceRule,
        quantity: SideTraceQuantity = "value",
        side: FieldTraceSide = "owner",
        runtime: FiniteElementRuntimeData | None = None,
    ) -> PreparedTraceAction:
        """Prepare the trace of one field on selected facets.

        See `prepare_finite_element_side_trace`.
        """

        # Lazy: fem._generic -> _point_interpolation -> _simplicial_locator -> _cell_map import cycle.
        from . import _point_interpolation as point_interpolation_owner

        return point_interpolation_owner.prepare_finite_element_side_trace(
            self,
            field_name,
            domain,
            rule=rule,
            quantity=quantity,
            side=side,
            runtime=runtime,
        )

    def reconstruct(
        self,
        field_name: str,
        coefficients: ArrayLike,
        block_name: str,
        reference_points: ArrayLike,
        /,
        *,
        runtime: FiniteElementRuntimeData | None = None,
    ) -> Array:
        field_index = self.field_index(field_name)
        block_index = self.dof_maps[field_index].block_names.index(str(block_name))
        points = jnp.asarray(reference_points)
        realized = self.default_runtime if runtime is None else runtime
        geometry = self.evaluate_block_geometry(
            field_name,
            block_index,
            realized.coordinates,
            points,
            jnp.ones((points.shape[0],), dtype=points.dtype),
        )
        local = jnp.asarray(coefficients)[
            self.dof_maps[field_index].cell_dofs[block_index]
        ]
        local = ein.contract(
            "cij,cj...->ci...",
            self.dof_maps[field_index].cell_transforms[block_index],
            local,
        )
        if geometry.basis_values.ndim == 2:
            return ein.contract("qi,ci...->cq...", geometry.basis_values, local)
        if geometry.basis_values.ndim == 3:
            return ein.contract("cqi,ci...->cq...", geometry.basis_values, local)
        return ein.contract("cqiv,ci->cqv", geometry.basis_values, local)

    def field_index(self, field_name: str, /) -> int:
        requested = str(field_name)
        for index, field_space in enumerate(self.field_spaces):
            if field_space.name == requested:
                return index
        raise KeyError(
            f"FiniteElementDiscretization.field_name must name a prepared field; got {requested!r}"
        )

    def _field_elements(self, field_index: int, /) -> tuple[FiniteElementSpec, ...]:
        return self.elements[field_index]

    def evaluate_geometry(
        self,
        field_name: str,
        coordinates: ArrayLike,
        /,
    ) -> tuple[FiniteElementBlockGeometry, ...]:
        field_index = self.field_index(field_name)
        points = jnp.asarray(coordinates)
        if points.shape != self.default_runtime.coordinates.shape:
            raise ValueError(
                "FiniteElementDiscretization.geometry.coordinates must preserve coordinate shape; got incompatible shape"
            )
        return tuple(
            _prepare_block_geometry(
                self.mesh,
                block,
                element,
                coordinates=points,
                coordinate_element=coordinate_element,
                geometry_dofs=geometry_dofs,
                precision_policy=self.precision_policy,
                coordinate_measure=self.coordinate_measure,
            )
            for block_index, (block, element, coordinate_element, geometry_dofs) in (
                enumerate(
                    zip(
                        self.mesh.blocks,
                        self.elements[field_index],
                        self.coordinate_elements,
                        self.coordinate_dofs,
                        strict=True,
                    )
                )
            )
        )

    def evaluate_block_geometry(
        self,
        field_name: str,
        block_index: int,
        coordinates: ArrayLike,
        reference_points: ArrayLike,
        reference_weights: ArrayLike,
        /,
    ) -> FiniteElementBlockGeometry:
        field_index = self.field_index(field_name)
        index = int(block_index)
        if index < 0 or index >= len(self.mesh.blocks):
            raise IndexError(
                f"FiniteElementDiscretization.block_index must index a mesh block; got {index}"
            )
        return _prepare_block_geometry(
            self.mesh,
            self.mesh.blocks[index],
            self.elements[field_index][index],
            coordinate_element=self.coordinate_elements[index],
            geometry_dofs=self.coordinate_dofs[index],
            coordinates=coordinates,
            reference_points=reference_points,
            reference_weights=reference_weights,
            precision_policy=self.precision_policy,
            coordinate_measure=self.coordinate_measure,
        )

    def _field_operator_geometry(
        self, field_name: str, runtime: FiniteElementRuntimeData, /
    ) -> tuple[FiniteElementDofMap, tuple[FiniteElementBlockGeometry, ...]]:
        if (
            runtime.topology_id != self.mesh.topology_id
            or runtime.geometry_layout_id != self.default_runtime.geometry_layout_id
        ):
            raise ValueError(
                "FiniteElementDiscretization.runtime must match topology, layout, and coordinate shape; got incompatible runtime metadata"
            )
        if (
            self.coordinate_measure is not None
            and runtime.coordinate_frame != self.coordinate_measure.frame
        ):
            raise ValueError(
                "FiniteElementDiscretization.runtime.coordinate_frame must match the coordinate measure; got a different frame"
            )
        field_index = self.field_index(field_name)
        geometries = (
            self.block_geometries[field_index]
            if runtime.runtime_id == self.default_runtime.runtime_id
            else self.evaluate_geometry(field_name, runtime.coordinates)
        )
        return self.dof_maps[field_index], geometries

    def _assemble_geometry_operator(
        self,
        dof_map: FiniteElementDofMap,
        geometries: tuple[FiniteElementBlockGeometry, ...],
        /,
        *,
        mass: bool,
    ) -> SparseLinearMap:
        tensors = geometry_local_tensors(geometries, mass=mass)
        return assemble_local_operator(
            dof_map,
            tensors,
            "finite-element-mass" if mass else "finite-element-stiffness",
            positive_definite=mass,
            component_shape=dof_map.component_shape,
            coefficient_dtype=self.coefficient_dtype,
        )

    @checked
    def assemble_field_operators(
        self, field_name: str, runtime: FiniteElementRuntimeData, /
    ) -> tuple[SparseLinearMap, SparseLinearMap]:
        dof_map, geometries = self._field_operator_geometry(field_name, runtime)
        return (
            self._assemble_geometry_operator(dof_map, geometries, mass=True),
            self._assemble_geometry_operator(dof_map, geometries, mass=False),
        )

    def assemble_cell_operator(
        self,
        field_name: str,
        local_values: Sequence[ArrayLike],
        /,
        *,
        operator_id: str,
        properties: OperatorProperties | None = None,
    ) -> SparseLinearMap:
        """Assemble fixed-topology cell matrices from local contributions."""
        field_index = self.field_index(field_name)
        values = tuple(jnp.asarray(value) for value in local_values)
        identifier = str(operator_id)
        if not identifier:
            raise ValueError(
                "FiniteElementDiscretization.cell_operator.operator_id must be nonempty; got an empty identifier"
            )
        if len(values) != len(self.mesh.blocks):
            raise ValueError(
                "FiniteElementDiscretization.cell_operator.local_values must contain one tensor per mesh block; got a different tensor count"
            )
        dof_map = self.dof_maps[field_index]
        for block_index, (block, value) in enumerate(
            zip(self.mesh.blocks, values, strict=True)
        ):
            width = dof_map.cell_dofs[block_index].shape[1]
            expected = (block.cell_count, width, width)
            if value.shape != expected:
                raise ValueError(
                    f"FiniteElementDiscretization.cell_operator.local_values must have shape {expected!r}; got {value.shape!r}"
                )
        properties_ = OperatorProperties() if properties is None else properties
        if not isinstance(properties_, OperatorProperties):
            raise TypeError(
                "FiniteElementDiscretization.cell_operator.properties must be OperatorProperties or None; got an incompatible object"
            )
        return assemble_local_operator(
            dof_map,
            values,
            identifier,
            positive_definite=False,
            properties=properties_,
            coefficient_dtype=self.coefficient_dtype,
        )


def evaluate_paired_field_basis(
    discretization: FiniteElementDiscretization,
    field_name: str,
    block_index: int | None,
    cells: ArrayLike,
    reference: ArrayLike,
    coordinates: ArrayLike,
    /,
    *,
    derivative_axis: int | None = None,
) -> tuple[Array, Array]:
    """Mapped basis for P paired cell/point queries, with O(P n) storage.

    Cell transforms are included once. A derivative
    is with respect to the requested physical coordinate, differentiating the
    full represented coordinate map and its Piola factors, including curved
    Jacobian/inverse-Jacobian/measure derivatives. No C-by-P cross product or
    P-by-P derivative Jacobian is constructed.
    """
    field_index = discretization.field_index(field_name)
    dof_map = discretization.dof_maps[field_index]
    index = 0 if block_index is None else block_index
    element = discretization.elements[field_index][index]
    coordinate_element = discretization.coordinate_elements[index]
    if block_index is None:
        if any(
            value.element_id != element.element_id
            for value in discretization.elements[field_index]
        ):
            raise ValueError(
                "finite_element.paired_queries.elements must be homogeneous across the whole support; got different field elements"
            )
        if any(
            value.element_id != coordinate_element.element_id
            for value in discretization.coordinate_elements
        ):
            raise ValueError(
                "finite_element.paired_queries.coordinate_elements must be homogeneous across the whole support; got different coordinate elements"
            )
        coordinate_routes = jnp.concatenate(discretization.coordinate_dofs, axis=0)
        transforms = jnp.concatenate(dof_map.cell_transforms, axis=0)
    else:
        coordinate_routes = discretization.coordinate_dofs[index]
        transforms = dof_map.cell_transforms[index]
    cell_count = coordinate_routes.shape[0]
    if not isinstance(
        coordinate_element,
        (
            FiniteElementSpec,
            cell_geometry_owner.BarycentricCellGeometryElement,
            cell_geometry_owner.RestrictedCellGeometryElement,
            cell_geometry_owner.PolynomialComposedCellGeometryElement,
            cell_geometry_owner.RationalComposedCellGeometryElement,
            cell_geometry_owner.SplineCellGeometryElement,
            cell_geometry_owner.LayerColumnCellGeometryElement,
        ),
    ):
        raise TypeError(
            "finite_element.paired_queries must use scalar coordinate tabulation; "
            "got an unsupported coordinate element"
        )
    indices = jnp.asarray(cells)
    policy = discretization.precision_policy
    points = policy.geometry(reference)
    coordinate_values = policy.geometry(coordinates)
    if indices.ndim != 1 or not jnp.issubdtype(indices.dtype, jnp.integer):
        raise ValueError(
            "finite_element.paired_queries.cells must be a one-dimensional integer array; got incompatible shape or dtype"
        )
    if points.shape != (indices.shape[0], element.topological_dimension):
        raise ValueError(
            "finite_element.paired_queries.points must contain one reference point per cell; got incompatible shape"
        )
    if derivative_axis is not None and not (
        isinstance(derivative_axis, int)
        and 0 <= derivative_axis < coordinate_values.shape[-1]
    ):
        raise ValueError(
            "finite_element.paired_queries.derivative_axis must identify a physical coordinate axis; got an invalid axis"
        )
    if cell_count == 0 or indices.size == 0:
        value_shape = (
            (coordinate_values.shape[-1],)
            if element.value_spec.proxy in ("circulation", "flux")
            else element.value_shape
        )
        return (
            jnp.zeros(
                (indices.size, element.local_dof_count) + value_shape,
                dtype=coordinate_values.dtype,
            ),
            jnp.zeros(indices.shape, dtype=jnp.bool_),
        )
    safe_cells = jnp.clip(indices, 0, cell_count - 1)
    finite_reference = jnp.all(jnp.isfinite(points), axis=-1)
    points = jnp.where(finite_reference[:, None], points, 0.0)
    routes = coordinate_routes[safe_cells]

    def mapped_values(
        selected_routes: Array,
        selected_points: Array,
        selected_cells: Array,
    ) -> tuple[Array, Array, Array]:
        _, jacobian, _, _, inverse, _, measure, _ = evaluate_coordinate_map(
            coordinate_element,
            selected_routes,
            coordinate_values,
            selected_points,
            precision_policy=policy,
            paired=True,
        )
        basis, _ = element.tabulate(selected_points)
        basis = map_reference_values(
            policy.evaluation(basis), element.value_spec, jacobian[:, None]
        )
        basis = ein.contract("pji,pj...->pi...", transforms[selected_cells], basis)
        return basis, inverse, measure

    if derivative_axis is None:
        basis, inverse, measure = mapped_values(routes, points, safe_cells)
        finite_basis = jnp.all(jnp.isfinite(basis).reshape((indices.size, -1)), axis=-1)
    else:

        def point_derivative(
            route: Array,
            point: Array,
            cell: Array,
        ) -> tuple[Array, tuple[Array, Array, Array]]:
            def single_basis(
                location: Array,
            ) -> tuple[Array, tuple[Array, Array, Array]]:
                values, inverse, measure = mapped_values(
                    route[None], location[None], cell[None]
                )
                return values[0], (values[0], inverse[0], measure[0])

            return jax.jacfwd(single_basis, has_aux=True)(point)

        reference_gradient, (basis, inverse, measure) = jax.vmap(point_derivative)(
            routes, points, safe_cells
        )
        finite_basis = jnp.all(jnp.isfinite(basis).reshape((indices.size, -1)), axis=-1)
        basis = ein.contract(
            "p...r,pr->p...", reference_gradient, inverse[..., derivative_axis]
        )
        finite_basis = finite_basis & jnp.all(
            jnp.isfinite(basis).reshape((indices.size, -1)), axis=-1
        )
    valid = (
        (indices >= 0)
        & (indices < cell_count)
        & finite_reference
        & jnp.isfinite(measure)
        & (measure > 0.0)
        & finite_basis
    )
    return (
        jnp.where(valid.reshape(valid.shape + (1,) * (basis.ndim - 1)), basis, 0.0),
        valid,
    )


@checked
def _prepare_block_geometry(
    mesh: cell_mesh_owner.CellMesh,
    block: CellBlock | PolyhedralBlock,
    element: FiniteElementSpec,
    /,
    *,
    coordinates: ArrayLike | None = None,
    reference_points: ArrayLike | None = None,
    reference_weights: ArrayLike | None = None,
    coordinate_element: CellGeometryElement | None = None,
    geometry_dofs: ArrayLike | None = None,
    precision_policy: FiniteElementPrecisionPolicy,
    coordinate_measure: CoordinateMeasurePlan | None = None,
    _preparation_bucket: bool = False,
) -> FiniteElementBlockGeometry:
    if (reference_points is None) != (reference_weights is None):
        raise ValueError(
            "finite_element.block_geometry.reference_quadrature must supply points and weights together; got one missing input"
        )
    match block:
        case CellBlock():
            pass
        case _:
            raise TypeError(
                "finite_element.block_geometry.block must be a fixed-cell block; got an incompatible block"
            )
    geometry_element = (
        cell_geometry_owner.coordinate_lagrange_element(block.cell_kind, 1)
        if coordinate_element is None
        else cell_geometry_owner.require_scalar_coordinate_element(
            coordinate_element, "finite_element.block_geometry.coordinate_element"
        )
    )
    if reference_points is None:
        points_, weights_ = degree_aware_reference_rule(
            block.cell_kind,
            max(_basis_cubature_degree(element), geometry_element.degree)
            + (
                1
                if coordinate_measure is not None
                and coordinate_measure.kind == "spherical-radial"
                else 0
            ),
        )
    else:
        points_ = jnp.asarray(reference_points)
        weights_ = jnp.asarray(reference_weights)
    routes = block.vertices if geometry_dofs is None else jnp.asarray(geometry_dofs)
    cell_count = block.cell_count
    bucket_count = (
        1 << (cell_count - 1).bit_length()
        if _preparation_bucket and cell_count
        else cell_count
    )
    if bucket_count != cell_count:
        routes = jax.device_put(
            np.pad(
                np.asarray(routes),
                ((0, bucket_count - cell_count), (0, 0)),
                mode="edge",
            )
        )
    (
        reference_points_,
        reference_weights_,
        physical_basis,
        reference_gradients,
        physical_points,
        physical_gradients,
        physical_weights,
        measure,
    ) = _evaluate_block_geometry(
        element,
        cell_geometry_owner.coordinate_tabulation(geometry_element),
        routes,
        mesh.coordinates if coordinates is None else jnp.asarray(coordinates),
        points_,
        weights_,
        precision_policy=precision_policy,
        coordinate_measure=coordinate_measure,
    )
    if bucket_count != cell_count:
        # Trim immutable preparation tables on the host: padding never enters
        # runtime fields or creates a separate XLA slice for every block width.
        physical_points = jax.device_put(np.asarray(physical_points)[:cell_count])
        physical_gradients = jax.device_put(np.asarray(physical_gradients)[:cell_count])
        physical_weights = jax.device_put(np.asarray(physical_weights)[:cell_count])
        measure = jax.device_put(np.asarray(measure)[:cell_count])
        if (
            element.mapping != "identity"
            or element.value_spec.form_type.twist != "untwisted"
        ):
            physical_basis = jax.device_put(np.asarray(physical_basis)[:cell_count])
    return FiniteElementBlockGeometry(
        block_name=block.name,
        reference_points=reference_points_,
        reference_weights=reference_weights_,
        basis_values=physical_basis,
        reference_gradients=reference_gradients,
        physical_points=physical_points,
        physical_gradients=physical_gradients,
        physical_weights=physical_weights,
        measure=measure,
        coordinate_measure=coordinate_measure,
    )


@eqx.filter_jit
def _evaluate_block_geometry(
    element: FiniteElementSpec,
    geometry_element: cell_geometry_owner.ScalarCoordinateTabulation,
    geometry_dofs: Array,
    coordinates: Array,
    points: Array,
    weights: Array,
    /,
    *,
    precision_policy: FiniteElementPrecisionPolicy,
    coordinate_measure: CoordinateMeasurePlan | None,
) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
    """Evaluate one geometry bucket without mesh, block, or epoch metadata."""
    reference_points = precision_policy.geometry(points)
    reference_weights = precision_policy.accumulation(weights)
    (
        physical_points,
        jacobian,
        _,
        inverse_metric,
        inverse_jacobian,
        _,
        measure_factor,
        _,
    ) = evaluate_coordinate_map(
        geometry_element,
        geometry_dofs,
        coordinates,
        reference_points,
        precision_policy=precision_policy,
        paired=False,
    )
    basis_values, reference_gradients = element.tabulate(reference_points)
    basis_values = precision_policy.evaluation(basis_values)
    reference_gradients = precision_policy.evaluation(reference_gradients)
    measure_factor = eqx.error_if(
        measure_factor,
        jnp.any(~jnp.isfinite(measure_factor) | (measure_factor <= 0.0)),
        "finite_element.block_geometry.metric must have positive finite metric determinant; got a nonpositive or nonfinite determinant",
    )
    if (
        element.mapping == "identity"
        and element.value_spec.form_type.twist == "untwisted"
    ):
        physical_basis = basis_values
        physical_gradients = ein.contract(
            "qkr,cqrd->cqkd",
            reference_gradients,
            inverse_jacobian,
        )
    else:
        physical_basis = map_reference_values(
            basis_values[None], element.value_spec, jacobian[:, :, None]
        )

        def mapped_reference(reference: Array, cell_coordinates: Array, /) -> Array:
            gradients = geometry_element.tabulate(reference[None])[1][0]
            jacobian_ = cell_coordinates.T @ gradients
            values_ = element.tabulate(reference[None])[0][0]
            return map_reference_values(values_, element.value_spec, jacobian_[None])

        mapped_gradient = jax.vmap(
            jax.vmap(jax.jacfwd(mapped_reference, argnums=0), in_axes=(0, None)),
            in_axes=(None, 0),
        )(reference_points, coordinates[geometry_dofs])
        if element.value_shape:
            physical_gradients = ein.contract(
                "cqivr,cqrd->cqivd", mapped_gradient, inverse_jacobian
            )
        else:
            physical_gradients = ein.contract(
                "cqir,cqrd->cqid", mapped_gradient, inverse_jacobian
            )
    physical_weights = precision_policy.accumulation(
        measure_factor * reference_weights[None, :]
    )
    if coordinate_measure is not None:
        physical_weights = coordinate_measure.weight(
            physical_points, physical_weights, frame=coordinate_measure.frame
        )
    if coordinate_measure is not None and coordinate_measure.radial_axis is not None:
        nodes = coordinates[geometry_dofs]
        valid = jnp.all(jnp.isfinite(nodes), axis=(-2, -1)) & jnp.all(
            nodes[..., coordinate_measure.radial_axis] >= 0.0, axis=-1
        )
        physical_weights = jnp.where(valid[:, None], physical_weights, jnp.nan)
    return (
        reference_points,
        reference_weights,
        physical_basis,
        reference_gradients,
        physical_points,
        physical_gradients,
        physical_weights,
        precision_policy.output(jnp.sum(physical_weights, axis=1)),
    )


def _basis_cubature_degree(element: FiniteElementSpec, /) -> int:
    if element.cell_kind == "pyramid":
        basis = element.form_basis
        if basis is not None:
            # Physical form components become polynomials on the collapsed
            # cube; Piola minors and its squared-height measure own extra work.
            return max(max(alpha) for alpha in basis.exponents) + 2
        return element.degree + 1
    return element.degree


@final
class MaskedFiniteElementPlan(StrictModule, NonTrainableState):
    """Static Lagrange finite-element plan for one `MaskedSimplexMesh` bucket.

    Every field is static: cell kind, ambient dimension, vertex and cell
    capacities, the layout ``signature_id``, family, degree, and precision policy.
    ``plan_id`` fingerprints the family, degree, layout signature, and precision
    policy only, so `assemble_masked_finite_element` compiles once per capacity
    bucket and serves every layout of that signature regardless of its active
    counts or topology values.

    Masked capacity layouts carry vertex DOFs only: DOF ``i`` is vertex slot
    ``i``, so the only admissible Lagrange degree is 1. Higher degrees need edge,
    face, and cell DOF slots the layout does not allocate and raise
    ``ValueError``.
    """

    cell_kind: str = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    vertex_capacity: int = eqx.field(static=True)
    cell_capacity: int = eqx.field(static=True)
    signature_id: str = eqx.field(static=True)
    family: str = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    precision_policy: FiniteElementPrecisionPolicy
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        mesh: MaskedSimplexMesh,
        /,
        *,
        degree: int = 1,
        precision_policy: FiniteElementPrecisionPolicy | None = None,
    ) -> None:
        if isinstance(degree, bool) or not isinstance(degree, (int, np.integer)):
            raise TypeError(
                "MaskedFiniteElementPlan.degree must be an integer; got a noninteger degree"
            )
        if degree != 1:
            raise ValueError(
                "MaskedFiniteElementPlan.degree must be 1 for vertex-only capacity layouts; "
                "got an unsupported Lagrange degree"
            )
        precision = (
            FiniteElementPrecisionPolicy()
            if precision_policy is None
            else precision_policy
        )
        self.cell_kind = mesh.cell_kind
        self.ambient_dimension = mesh.ambient_dimension
        self.vertex_capacity = mesh.vertex_capacity
        self.cell_capacity = mesh.cell_capacity
        self.signature_id = mesh.signature_id
        self.family = "lagrange"
        self.degree = 1
        self.precision_policy = precision
        self.plan_id = canonical_fingerprint(
            {
                "kind": "masked-finite-element-plan",
                "family": "lagrange",
                "degree": 1,
                "mesh_signature": mesh.signature_id,
                "precision_policy": precision.policy_id,
            }
        )


@final
class MaskedFiniteElementSystem(StrictModule, NonTrainableState):
    """Capacity-shaped P1 mass and stiffness operators of one masked layout.

    Both operators act on ``(vertex_capacity,)`` vectors and share one
    `EdgeRelation`: ``cell_capacity * (d + 1) ** 2`` cell routes valid only on
    active cells, then ``vertex_capacity`` diagonal routes valid only on inactive
    vertex slots with coefficient 1. Inactive DOFs are therefore pinned through
    identity rows: each operator stays square, is the identity on padding, and
    never couples padding to active DOFs, so solve, transpose, and adjoint
    actions keep the capacity shape and map zero padding to zero padding.
    ``mass`` is symmetric positive definite; ``stiffness`` is symmetric positive
    semidefinite with the constants of each active component as its kernel.
    ``boundary_dofs`` marks active vertices lying on a boundary facet. Operator
    IDs derive from ``plan_id`` alone, never from topology values.
    """

    plan_id: str = eqx.field(static=True)
    mass: SparseLinearMap
    stiffness: SparseLinearMap
    dof_active: Array
    boundary_dofs: Array

    @checked
    def __init__(
        self,
        plan_id: str,
        mass: SparseLinearMap,
        stiffness: SparseLinearMap,
        dof_active: Array,
        boundary_dofs: Array,
        /,
    ) -> None:
        if not isinstance(plan_id, str) or not plan_id:
            raise ValueError(
                "MaskedFiniteElementSystem.plan_id must be a nonempty string; got an incompatible or empty identifier"
            )
        shape = mass.input_shape
        if (
            mass.output_shape != shape
            or stiffness.input_shape != shape
            or stiffness.output_shape != shape
        ):
            raise ValueError(
                "MaskedFiniteElementSystem.operators must share one square DOF space; got incompatible operator shapes"
            )
        for name, mask in (("dof_active", dof_active), ("boundary_dofs", boundary_dofs)):
            if mask.dtype != jnp.bool_ or mask.shape != shape:
                raise TypeError(
                    f"MaskedFiniteElementSystem.{name} must be a boolean {shape} array; got dtype {mask.dtype}, shape {mask.shape}"
                )
        self.plan_id = plan_id
        self.mass = mass
        self.stiffness = stiffness
        self.dof_active = dof_active
        self.boundary_dofs = boundary_dofs


def _masked_local_tensors(
    plan: MaskedFiniteElementPlan, mesh: MaskedSimplexMesh, /
) -> tuple[Array, Array]:
    """P1 local mass and stiffness on every cell lane, exactly zero on padding.

    Inactive lanes are mapped onto the reference simplex, appended after the
    vertex slots, so the metric inverse and the positive-measure check never see
    padding rows; their finite tensors are then replaced by exact zeros.
    """

    element = lagrange_element(plan.cell_kind, plan.degree)
    policy = plan.precision_policy
    width = mesh.dimension + 1
    reference_vertices = jnp.pad(
        jnp.asarray(element.reference_nodes, dtype=mesh.coordinates.dtype),
        ((0, 0), (0, plan.ambient_dimension - mesh.dimension)),
    )
    reference_slots = jnp.arange(
        plan.vertex_capacity, plan.vertex_capacity + width, dtype=jnp.int32
    )
    routes = jnp.where(mesh.cell_active[:, None], mesh.cells, reference_slots[None, :])
    points_, weights_ = degree_aware_reference_rule(plan.cell_kind, plan.degree)
    reference_points = policy.geometry(points_)
    reference_weights = policy.accumulation(weights_)
    physical_points, jacobian, _, inverse_metric, _, _, measure_factor, _ = (
        evaluate_coordinate_map(
            element,
            routes,
            jnp.concatenate((mesh.coordinates, reference_vertices)),
            reference_points,
            precision_policy=policy,
            paired=False,
        )
    )
    measure_factor = eqx.error_if(
        measure_factor,
        jnp.any(~jnp.isfinite(measure_factor) | (measure_factor <= 0.0)),
        "MaskedFiniteElementSystem.active_cell_metric must have positive finite metric determinants; "
        "got a nonpositive or nonfinite determinant on an active cell",
    )
    basis_values, reference_gradients = element.tabulate(reference_points)
    reference_gradients = policy.evaluation(reference_gradients)
    physical_weights = policy.accumulation(measure_factor * reference_weights[None, :])
    geometry = FiniteElementBlockGeometry(
        block_name=plan.cell_kind,
        reference_points=reference_points,
        reference_weights=reference_weights,
        basis_values=policy.evaluation(basis_values),
        reference_gradients=reference_gradients,
        physical_points=physical_points,
        physical_gradients=ein.contract(
            "cqdi,cqij,qkj->cqkd", jacobian, inverse_metric, reference_gradients
        ),
        physical_weights=physical_weights,
        measure=policy.output(jnp.sum(physical_weights, axis=1)),
    )
    active = mesh.cell_active[:, None, None]
    zero = jnp.zeros((), dtype=physical_weights.dtype)
    return (
        jnp.where(active, local_mass_tensor(geometry), zero),
        jnp.where(active, local_stiffness_tensor(geometry), zero),
    )


def _masked_boundary_dofs(mesh: MaskedSimplexMesh, /) -> Array:
    """Active vertex slots on a boundary facet.

    Local vertex ``j`` lies on the facet opposite local vertex ``i`` iff
    ``i != j``; per-cell flags reduce onto vertex slots over the active
    cell-vertex incidence relation.
    """

    width = mesh.dimension + 1
    opposite = jnp.asarray(1 - np.eye(width, dtype=np.int32))
    on_boundary = mesh.boundary_facets.astype(jnp.int32) @ opposite
    incidence = EdgeRelation(
        jnp.arange(mesh.cell_capacity * width, dtype=jnp.int32),
        mesh.cells.reshape((-1,)),
        source_size=mesh.cell_capacity * width,
        target_size=mesh.vertex_capacity,
        valid=jnp.repeat(mesh.cell_active, width),
    )
    counts = route_reduce(incidence, on_boundary.reshape((-1,)))
    return (counts > 0) & mesh.vertex_active


@eqx.filter_jit
def assemble_masked_finite_element(
    plan: MaskedFiniteElementPlan, mesh: MaskedSimplexMesh, /
) -> MaskedFiniteElementSystem:
    """Assemble capacity-shaped P1 mass and stiffness on one masked layout.

    Module-level compiled entry. Its compile key is the plan's static identity
    plus the layout signature (cell kind, ambient dimension, capacities,
    coordinate dtype); coordinates, IDs, masks, cells, and adjacency are traced,
    so layouts of one bucket with different active counts or topology reuse one
    executable. See `MaskedFiniteElementSystem` for the pinned-padding layout.
    """

    if not isinstance(plan, MaskedFiniteElementPlan):
        raise TypeError(
            "assemble_masked_finite_element.plan must be a MaskedFiniteElementPlan; got an incompatible object"
        )
    if not isinstance(mesh, MaskedSimplexMesh):
        raise TypeError(
            "assemble_masked_finite_element.mesh must be a MaskedSimplexMesh; got an incompatible object"
        )
    if mesh.signature_id != plan.signature_id:
        raise ValueError(
            "assemble_masked_finite_element.mesh.signature_id must match the plan capacity bucket; got a different signature"
        )
    width = mesh.dimension + 1
    mass_local, stiffness_local = _masked_local_tensors(plan, mesh)
    columns = jnp.broadcast_to(mesh.cells[:, None, :], (plan.cell_capacity, width, width))
    slots = jnp.arange(plan.vertex_capacity, dtype=jnp.int32)
    relation = EdgeRelation(
        jnp.concatenate((columns.reshape((-1,)), slots)),
        jnp.concatenate((jnp.swapaxes(columns, 1, 2).reshape((-1,)), slots)),
        source_size=plan.vertex_capacity,
        target_size=plan.vertex_capacity,
        valid=jnp.concatenate(
            (jnp.repeat(mesh.cell_active, width * width), ~mesh.vertex_active)
        ),
    )
    pins = jnp.ones((plan.vertex_capacity,), dtype=mass_local.dtype)
    mass = SparseLinearMap(
        relation,
        jnp.concatenate((mass_local.reshape((-1,)), pins)),
        properties=OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            positive_semidefinite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_definite": "construction",
                "positive_semidefinite": "construction",
            },
        ),
        operator_id=canonical_fingerprint(
            {"kind": "masked-finite-element-mass", "plan": plan.plan_id}
        ),
    )
    stiffness = SparseLinearMap(
        relation,
        jnp.concatenate((stiffness_local.reshape((-1,)), pins)),
        properties=OperatorProperties(
            self_adjoint=True,
            positive_semidefinite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_semidefinite": "construction",
            },
        ),
        operator_id=canonical_fingerprint(
            {"kind": "masked-finite-element-stiffness", "plan": plan.plan_id}
        ),
    )
    return MaskedFiniteElementSystem(
        plan.plan_id,
        mass,
        stiffness,
        mesh.vertex_active,
        _masked_boundary_dofs(mesh),
    )


@checked
def constrain_masked_dofs(
    operator: SparseLinearMap, pinned: ArrayLike, /
) -> SparseLinearMap:
    """Pin DOFs of a square edge-relation operator through identity rows.

    Returns ``P_free A P_free + P_pinned``: routes touching a pinned source or
    target are invalidated and one appended diagonal route per DOF, valid only
    where ``pinned``, carries 1. The route capacity is static (operator routes
    plus one per DOF), so a traced ``pinned`` mask never changes the compiled
    shape. With ``pinned = system.boundary_dofs`` a masked stiffness becomes the
    homogeneous Dirichlet operator; padding stays pinned whether or not it is
    included in ``pinned``. Self-adjointness and (semi)definiteness carry over
    as transformed evidence; the operator ID derives from the input operator ID.
    """

    match operator.relation:
        case EdgeRelation() as relation:
            pass
        case _:
            raise TypeError(
                "constrain_masked_dofs.operator.relation must be an EdgeRelation; got an incompatible relation"
            )
    if operator.batch_shape or relation.source_size != relation.target_size:
        raise ValueError(
            "constrain_masked_dofs.operator must be unbatched and square; got an incompatible operator shape"
        )
    mask = jnp.asarray(pinned)
    if mask.dtype != jnp.bool_:
        raise TypeError(
            "constrain_masked_dofs.pinned must be a boolean array; got a nonboolean dtype"
        )
    if mask.shape != (relation.source_size,):
        raise ValueError(
            "constrain_masked_dofs.pinned must hold one flag per DOF; got incompatible shape"
        )
    slots = jnp.arange(relation.source_size, dtype=relation.source_indices.dtype)
    free = (
        relation.valid
        & ~gather_routes(relation, mask)
        & ~gather_routes(relation.transpose(), mask)
    )
    constrained = EdgeRelation(
        jnp.concatenate((relation.source_indices, slots)),
        jnp.concatenate((relation.target_indices, slots)),
        source_size=relation.source_size,
        target_size=relation.target_size,
        valid=jnp.concatenate((free, mask)),
    )
    source = operator.properties
    claims = {
        "self_adjoint": source.self_adjoint,
        "positive_definite": source.positive_definite,
        "positive_semidefinite": source.positive_semidefinite,
    }
    return SparseLinearMap(
        constrained,
        jnp.concatenate(
            (
                operator.coefficients,
                jnp.ones((relation.source_size,), dtype=operator.coefficients.dtype),
            )
        ),
        properties=OperatorProperties(
            **claims,
            evidence={
                name: "transformed"
                for name, claimed in claims.items()
                if claimed and source.evidence_for(name) != "unknown"
            },
        ),
        operator_id=canonical_fingerprint(
            {"kind": "masked-dof-constraint", "operator": operator.operator_id}
        ),
    )


__all__ = [
    "FiniteElementDiscretization",
    "FiniteElementDofMap",
    "FiniteElementFieldSpec",
    "FiniteElementPlan",
    "FiniteElementRuntimeData",
    "IntegrationDomain",
    "MaskedFiniteElementPlan",
    "MaskedFiniteElementSystem",
    "assemble_masked_finite_element",
    "constrain_masked_dofs",
]
