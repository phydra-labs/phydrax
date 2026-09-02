#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from math import prod

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    ArraySpace,
    BlockSpace,
    inverse_small_linear,
    OperatorProperties,
    SmallLinearSolvePlan,
)
from ...sparse import EdgeRelation, RowRelation, SparseLinearMap
from .._cell_complex import (
    IntervalConnectivity,
    PolygonalConnectivity,
    PolyhedralConnectivity,
)
from .._cell_mesh import CellBlock, CellMesh
from .._core import (
    DiscretizationCapability,
    DiscretizationKey,
    DiscretizationRole,
    PreparationReport,
)
from .._hexahedral import (
    _EDGES as _HEXAHEDRAL_EDGES,
    _FACES as _HEXAHEDRAL_FACES,
    _quadrilateral_tensor_permutation,
    HexahedralConnectivity,
)
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
from .._spaces import BlockDofLayout, DiscreteFieldSpace, EntityDofLayout
from .._support import DiscreteSupport
from .._topology import EntitySelection
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
        if len({element.conformity for element in resolved}) != 1:
            raise ValueError("One field must use one conformity across mesh blocks.")
        if len({element.representation for element in resolved}) != 1:
            raise ValueError(
                "One field must use one coefficient representation across mesh blocks."
            )
        if len({element.mapping for element in resolved}) != 1:
            raise ValueError("One field must use one mapping across mesh blocks.")
        if len({element.value_shape for element in resolved}) != 1:
            raise ValueError("One field must use one value shape across mesh blocks.")
        return resolved


class FiniteElementCoordinateSpec(StrictModule, NonTrainableState):
    """Per-block coordinate elements, geometry DOF routes, and default coordinates."""

    block_names: tuple[str, ...] = eqx.field(static=True)
    elements: tuple[FiniteElementSpec, ...]
    geometry_dofs: tuple[Array, ...]
    coordinates: Array
    coordinate_spec_id: str = eqx.field(static=True)

    def __init__(
        self,
        elements: Mapping[str, FiniteElementSpec],
        geometry_dofs: Mapping[str, ArrayLike],
        coordinates: ArrayLike,
        /,
    ):
        items = tuple(sorted((str(name), element) for name, element in elements.items()))
        routes = {
            str(name): np.asarray(value, dtype=np.int32)
            for name, value in geometry_dofs.items()
        }
        points = np.asarray(coordinates, dtype=float)
        if not items or any(not name for name, _ in items):
            raise ValueError("Coordinate element mapping must be non-empty.")
        if not all(isinstance(element, FiniteElementSpec) for _, element in items):
            raise TypeError("Coordinate elements must be FiniteElementSpec values.")
        if any(element.conformity != "H1" for _, element in items):
            raise ValueError("Coordinate elements must be H1-conforming.")
        if set(routes) != {name for name, _ in items}:
            raise ValueError(
                "Coordinate DOF routes must match coordinate element blocks."
            )
        if points.ndim != 2 or not np.all(np.isfinite(points)):
            raise ValueError("Coordinate values must be one finite rank-2 array.")
        normalized_routes = []
        for name, element in items:
            route = routes[name]
            if route.ndim != 2 or route.shape[1] != element.local_dof_count:
                raise ValueError(
                    "Coordinate DOF route width must match its coordinate element."
                )
            if np.any(route < 0) or np.any(route >= points.shape[0]):
                raise ValueError("Coordinate DOF routes index undeclared coordinates.")
            normalized_routes.append(jnp.asarray(route))
        self.block_names = tuple(name for name, _ in items)
        self.elements = tuple(element for _, element in items)
        self.geometry_dofs = tuple(normalized_routes)
        self.coordinates = jnp.asarray(points)
        self.coordinate_spec_id = canonical_fingerprint(
            {
                "kind": "finite-element-coordinate-spec",
                "blocks": [[name, element.element_id] for name, element in items],
                "geometry_dofs": [
                    array_tree_fingerprint(np.asarray(value))
                    for value in normalized_routes
                ],
                "coordinate_shape": list(points.shape),
            }
        )

    @classmethod
    def affine(cls, mesh: CellMesh, /) -> FiniteElementCoordinateSpec:
        return cls(
            {block.name: lagrange_element(block.cell_kind, 1) for block in mesh.blocks},
            {block.name: block.vertices for block in mesh.blocks},
            mesh.coordinates,
        )

    def resolve(
        self,
        mesh: CellMesh,
        /,
    ) -> tuple[
        tuple[FiniteElementSpec, ...],
        tuple[Array, ...],
        Array,
    ]:
        mapping = dict(zip(self.block_names, self.elements, strict=True))
        routes = dict(zip(self.block_names, self.geometry_dofs, strict=True))
        if set(mapping) != {block.name for block in mesh.blocks}:
            raise ValueError("Coordinate element assignments must match mesh blocks.")
        resolved = tuple(mapping[block.name] for block in mesh.blocks)
        resolved_routes = tuple(routes[block.name] for block in mesh.blocks)
        for block, element, route in zip(
            mesh.blocks,
            resolved,
            resolved_routes,
            strict=True,
        ):
            if block.cell_kind != element.cell_kind:
                raise ValueError("Coordinate element cell kind does not match its block.")
            if route.shape[0] != block.cell_count:
                raise ValueError("Coordinate DOF routes require one row per cell.")
        return resolved, resolved_routes, self.coordinates


_HEXAHEDRAL_EDGE_BY_VERTICES = {
    frozenset(edge): index for index, edge in enumerate(_HEXAHEDRAL_EDGES)
}


def _has_nonvertex_dofs(element: FiniteElementSpec, /) -> bool:
    return any(
        entity
        for dimension_entities in element.entity_dofs[1:]
        for entity in dimension_entities
    )


def _hexahedral_face_shape(
    element: FiniteElementSpec,
    local_face: int,
    /,
) -> tuple[int, int]:
    vertices = _HEXAHEDRAL_FACES[local_face]
    side_edges = tuple(
        _HEXAHEDRAL_EDGE_BY_VERTICES[
            frozenset((vertices[position], vertices[(position + 1) % 4]))
        ]
        for position in range(4)
    )
    side_widths = tuple(
        len(element.entity_dofs[1][local_edge]) for local_edge in side_edges
    )
    if side_widths[0] != side_widths[2] or side_widths[1] != side_widths[3]:
        raise ValueError(
            "Hexahedral tensor faces require equal widths on opposite edges."
        )
    shape = (side_widths[0], side_widths[3])
    if len(element.entity_dofs[2][local_face]) != prod(shape):
        raise ValueError(
            "Hexahedral face-interior DOFs do not match its tensor trace shape."
        )
    return shape


def _canonical_face_shape(
    vertex_permutation: ArrayLike,
    local_shape: tuple[int, int],
    /,
) -> tuple[int, int]:
    permutation = np.asarray(vertex_permutation, dtype=np.int32)
    corners = np.asarray(((0, 0), (1, 0), (1, 1), (0, 1)), dtype=np.int32)
    direction_u = corners[permutation[1]] - corners[permutation[0]]
    if direction_u[0] != 0:
        return local_shape
    return local_shape[1], local_shape[0]


def _hexahedral_face_grid_positions(
    element: FiniteElementSpec,
    local_face: int,
    shape: tuple[int, int],
    /,
) -> np.ndarray:
    face_dofs = element.entity_dofs[2][local_face]
    if not face_dofs:
        return np.empty((0,), dtype=np.int32)
    face_vertices = _HEXAHEDRAL_FACES[local_face]
    vertex_dofs = element.entity_dofs[0]
    if any(len(vertex_dofs[vertex]) != 1 for vertex in face_vertices):
        raise ValueError("H1 nodal vertices require one DOF per vertex.")
    nodes = np.asarray(element.reference_nodes, dtype=float)
    corners = nodes[
        np.asarray(
            [vertex_dofs[vertex][0] for vertex in face_vertices],
            dtype=np.int32,
        )
    ]
    origin = corners[0]
    axes = np.stack((corners[1] - origin, corners[3] - origin), axis=1)
    parameters = np.linalg.lstsq(
        axes,
        nodes[np.asarray(face_dofs, dtype=np.int32)].T - origin[:, None],
        rcond=None,
    )[0].T
    reconstructed = origin + parameters @ axes.T
    if not np.allclose(
        reconstructed,
        nodes[np.asarray(face_dofs, dtype=np.int32)],
        rtol=1.0e-10,
        atol=1.0e-12,
    ):
        raise ValueError("Hexahedral face DOFs are not on their declared face.")
    parameters = np.round(parameters, decimals=12)
    levels_u = np.unique(parameters[:, 0])
    levels_v = np.unique(parameters[:, 1])
    if (len(levels_u), len(levels_v)) != shape:
        raise ValueError(
            "Hexahedral face DOF coordinates do not match its tensor trace shape."
        )
    positions = np.empty((len(face_dofs),), dtype=np.int32)
    for position, (parameter_u, parameter_v) in enumerate(parameters):
        index_u = int(np.argmin(np.abs(levels_u - parameter_u)))
        index_v = int(np.argmin(np.abs(levels_v - parameter_v)))
        if not np.isclose(levels_u[index_u], parameter_u) or not np.isclose(
            levels_v[index_v], parameter_v
        ):
            raise ValueError("Hexahedral face tensor coordinates are inconsistent.")
        positions[position] = index_u * shape[1] + index_v
    if np.unique(positions).size != len(face_dofs):
        raise ValueError("Hexahedral face tensor positions must be unique.")
    return positions


def _uniform_entity_width(widths: np.ndarray, /) -> int:
    unique = np.unique(widths)
    return int(unique[0]) if unique.size == 1 and unique[0] > 0 else 1


class FiniteElementDofMap(StrictModule, NonTrainableState):
    """Per-block FE local gathers into one global field coordinate array."""

    block_names: tuple[str, ...] = eqx.field(static=True)
    cell_dofs: tuple[Array, ...]
    relations: tuple[RowRelation, ...]
    orientations: tuple[Array, ...]
    cell_coordinate_weights: tuple[Array, ...]
    global_dof_count: int = eqx.field(static=True)
    entity_dof_counts: tuple[int, ...] = eqx.field(static=True)
    entity_dofs_per_entity: tuple[int, ...] = eqx.field(static=True)
    component_shape: tuple[int, ...] = eqx.field(static=True)
    association: str = eqx.field(static=True)
    boundary_dof_mask: Array
    dof_coordinates: Array
    dof_map_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh: CellMesh,
        elements: Sequence[FiniteElementSpec],
        /,
        *,
        component_shape: Sequence[int] = (),
    ):
        resolved = tuple(elements)
        if len(resolved) != len(mesh.blocks):
            raise ValueError("One finite element is required per mesh block.")
        components = tuple(int(size) for size in component_shape)
        if any(size <= 0 for size in components):
            raise ValueError("DOF component dimensions must be positive.")
        conformities = {element.conformity for element in resolved}
        if len(conformities) != 1:
            raise ValueError("One field must use one conformity across cell blocks.")
        conformity = conformities.pop()
        vertex_count = int(mesh.coordinates.shape[0])
        connectivity = mesh.connectivity
        topological_dimension = mesh.topological_dimension
        entity_dof_counts = (0,) * (topological_dimension + 1)
        entity_dofs_per_entity = (1,) * (topological_dimension + 1)
        edge_widths = None
        edge_starts = None
        face_shapes = None
        face_starts = None
        cell_widths = None
        cell_starts = None

        if conformity == "L2":
            association = "cell"
            global_count = sum(
                block.cell_count * element.local_dof_count
                for block, element in zip(mesh.blocks, resolved, strict=True)
            )
        elif conformity in ("Hdiv", "Hcurl"):
            if not isinstance(connectivity, PolygonalConnectivity):
                raise ValueError(
                    "Compatible edge spaces currently require polygonal mesh."
                )
            if components:
                raise ValueError(
                    "H(div)/H(curl) element values cannot add replicated components."
                )
            association = "edge"
            global_count = int(connectivity.edges.shape[0])
        elif conformity == "H1":
            high_order = any(_has_nonvertex_dofs(element) for element in resolved)
            if not high_order:
                association = "vertex"
                global_count = vertex_count
            else:
                if not isinstance(
                    connectivity,
                    (PolygonalConnectivity, HexahedralConnectivity),
                ):
                    raise ValueError(
                        "High-order H1 entity routing requires polygonal or "
                        "hexahedral connectivity."
                    )
                association = "entity"
                edge_count = int(connectivity.edges.shape[0])
                edge_widths = np.full((edge_count,), -1, dtype=np.int32)
                total_cell_count = sum(block.cell_count for block in mesh.blocks)
                cell_widths = np.empty((total_cell_count,), dtype=np.int32)
                if isinstance(connectivity, HexahedralConnectivity):
                    face_count = int(connectivity.faces.shape[0])
                    face_shapes = np.full((face_count, 2), -1, dtype=np.int32)

                cell_offset = 0
                for block, element in zip(
                    mesh.blocks,
                    resolved,
                    strict=True,
                ):
                    if len(element.entity_dofs[0]) != block.arity:
                        raise ValueError(
                            "H1 nodal vertex entities must match the cell vertices."
                        )
                    local_edge_count = len(element.entity_dofs[1])
                    expected_edge_count = (
                        block.arity
                        if isinstance(connectivity, PolygonalConnectivity)
                        else len(_HEXAHEDRAL_EDGES)
                    )
                    if local_edge_count != expected_edge_count:
                        raise ValueError(
                            "H1 edge entities must match the reference cell edges."
                        )
                    block_cell_edges = np.asarray(
                        connectivity.cell_edges,
                        dtype=np.int32,
                    )[
                        cell_offset : cell_offset + block.cell_count,
                        :local_edge_count,
                    ]
                    for local_edge, edge_dofs in enumerate(element.entity_dofs[1]):
                        width = len(edge_dofs)
                        for edge in np.unique(block_cell_edges[:, local_edge]):
                            existing = edge_widths[int(edge)]
                            if existing >= 0 and existing != width:
                                raise ValueError(
                                    "Shared H1 edge trace widths are incompatible; "
                                    "a mortar is required."
                                )
                            edge_widths[int(edge)] = width

                    if isinstance(connectivity, HexahedralConnectivity):
                        if face_shapes is None:
                            raise RuntimeError(
                                "Hexahedral H1 routing requires allocated face shapes."
                            )
                        if len(element.entity_dofs[2]) != len(_HEXAHEDRAL_FACES):
                            raise ValueError(
                                "H1 face entities must match the hexahedron faces."
                            )
                        block_cell_faces = np.asarray(
                            connectivity.cell_faces,
                            dtype=np.int32,
                        )[cell_offset : cell_offset + block.cell_count]
                        block_face_permutations = np.asarray(
                            connectivity.cell_face_vertex_permutations,
                            dtype=np.int32,
                        )[cell_offset : cell_offset + block.cell_count]
                        for local_face in range(len(_HEXAHEDRAL_FACES)):
                            local_shape = _hexahedral_face_shape(
                                element,
                                local_face,
                            )
                            for cell in range(block.cell_count):
                                face = int(block_cell_faces[cell, local_face])
                                canonical_shape = _canonical_face_shape(
                                    block_face_permutations[cell, local_face],
                                    local_shape,
                                )
                                existing = tuple(
                                    int(value) for value in face_shapes[face]
                                )
                                if existing[0] >= 0 and existing != canonical_shape:
                                    raise ValueError(
                                        "Shared H1 quadrilateral trace shapes are "
                                        "incompatible; a mortar is required."
                                    )
                                face_shapes[face] = canonical_shape

                    top_entities = element.entity_dofs[topological_dimension]
                    if len(top_entities) != 1:
                        raise ValueError(
                            "H1 cell interiors require one top-dimensional entity."
                        )
                    cell_widths[cell_offset : cell_offset + block.cell_count] = len(
                        top_entities[0]
                    )
                    cell_offset += block.cell_count

                if np.any(edge_widths < 0):
                    raise ValueError("High-order H1 routing left unassigned edges.")
                if face_shapes is not None and np.any(face_shapes < 0):
                    raise ValueError("High-order H1 routing left unassigned faces.")

                cursor = vertex_count
                edge_starts = np.empty_like(edge_widths)
                for edge, width in enumerate(edge_widths):
                    edge_starts[edge] = cursor
                    cursor += int(width)
                edge_dof_count = cursor - vertex_count

                face_dof_count = 0
                if face_shapes is not None:
                    face_starts = np.empty((len(face_shapes),), dtype=np.int32)
                    for face, shape in enumerate(face_shapes):
                        face_starts[face] = cursor
                        cursor += int(prod(tuple(int(value) for value in shape)))
                    face_dof_count = cursor - vertex_count - edge_dof_count

                cell_starts = np.empty_like(cell_widths)
                cell_global_ids = np.concatenate(
                    tuple(
                        np.asarray(block.global_ids, dtype=np.int64)
                        for block in mesh.blocks
                    )
                )
                for cell in np.argsort(cell_global_ids, kind="stable"):
                    cell_starts[cell] = cursor
                    cursor += int(cell_widths[cell])
                cell_dof_count = cursor - (vertex_count + edge_dof_count + face_dof_count)
                global_count = cursor

                counts = [vertex_count, edge_dof_count]
                per_entity = [1, _uniform_entity_width(edge_widths)]
                if topological_dimension == 3:
                    if face_shapes is None:
                        raise TypeError("Hexahedral H1 routing requires face shapes.")
                    face_widths = np.prod(face_shapes, axis=1)
                    counts.append(face_dof_count)
                    per_entity.append(_uniform_entity_width(face_widths))
                counts.append(cell_dof_count)
                per_entity.append(_uniform_entity_width(cell_widths))
                entity_dof_counts = tuple(counts)
                entity_dofs_per_entity = tuple(per_entity)
        else:
            raise ValueError(f"Unsupported finite-element conformity {conformity!r}.")

        block_dofs = []
        orientations = []
        relations = []
        cell_offset = 0
        dof_offset = 0
        for block, element in zip(mesh.blocks, resolved, strict=True):
            vertices = np.asarray(block.vertices, dtype=np.int32)
            if association == "cell":
                width = element.local_dof_count
                local = np.arange(
                    dof_offset,
                    dof_offset + block.cell_count * width,
                    dtype=np.int32,
                ).reshape((block.cell_count, width))
                dof_offset += block.cell_count * width
                orientation = np.ones_like(local, dtype=float)
            elif association == "edge":
                if not isinstance(connectivity, PolygonalConnectivity):
                    raise TypeError(
                        "Compatible edge map requires polygonal connectivity."
                    )
                local = np.asarray(connectivity.cell_edges, dtype=np.int32)[
                    cell_offset : cell_offset + block.cell_count,
                    : element.local_dof_count,
                ]
                orientation = np.asarray(
                    connectivity.cell_edge_signs,
                    dtype=float,
                )[
                    cell_offset : cell_offset + block.cell_count,
                    : element.local_dof_count,
                ]
            elif association == "entity":
                if not isinstance(
                    connectivity,
                    (PolygonalConnectivity, HexahedralConnectivity),
                ):
                    raise RuntimeError(
                        "High-order H1 routing lost compatible connectivity."
                    )
                if edge_widths is None or edge_starts is None or cell_starts is None:
                    raise TypeError("High-order H1 entity offsets are unavailable.")
                local = np.full(
                    (block.cell_count, element.local_dof_count),
                    -1,
                    dtype=np.int32,
                )
                for local_vertex, entity_dofs in enumerate(element.entity_dofs[0]):
                    if len(entity_dofs) != 1:
                        raise ValueError("H1 nodal vertices require one DOF per vertex.")
                    local[:, entity_dofs[0]] = vertices[:, local_vertex]

                local_edge_count = len(element.entity_dofs[1])
                block_cell_edges = np.asarray(
                    connectivity.cell_edges,
                    dtype=np.int32,
                )[
                    cell_offset : cell_offset + block.cell_count,
                    :local_edge_count,
                ]
                block_cell_signs = np.asarray(
                    connectivity.cell_edge_signs,
                    dtype=float,
                )[
                    cell_offset : cell_offset + block.cell_count,
                    :local_edge_count,
                ]
                for local_edge, entity_dofs in enumerate(element.entity_dofs[1]):
                    width = len(entity_dofs)
                    if width == 0:
                        continue
                    positions = np.arange(width, dtype=np.int32)
                    canonical_positions = np.where(
                        block_cell_signs[:, local_edge, None] > 0.0,
                        positions,
                        positions[::-1],
                    )
                    local[:, np.asarray(entity_dofs, dtype=np.int32)] = (
                        edge_starts[block_cell_edges[:, local_edge], None]
                        + canonical_positions
                    )

                if isinstance(connectivity, HexahedralConnectivity):
                    if face_starts is None:
                        raise TypeError("Hexahedral H1 face offsets are unavailable.")
                    block_cell_faces = np.asarray(
                        connectivity.cell_faces,
                        dtype=np.int32,
                    )[cell_offset : cell_offset + block.cell_count]
                    block_face_permutations = np.asarray(
                        connectivity.cell_face_vertex_permutations,
                        dtype=np.int32,
                    )[cell_offset : cell_offset + block.cell_count]
                    for local_face, face_dofs in enumerate(element.entity_dofs[2]):
                        if not face_dofs:
                            continue
                        shape = _hexahedral_face_shape(element, local_face)
                        grid_positions = _hexahedral_face_grid_positions(
                            element,
                            local_face,
                            shape,
                        )
                        for cell in range(block.cell_count):
                            tensor_permutation = _quadrilateral_tensor_permutation(
                                block_face_permutations[cell, local_face],
                                *shape,
                            )
                            face = block_cell_faces[cell, local_face]
                            local[
                                cell,
                                np.asarray(face_dofs, dtype=np.int32),
                            ] = face_starts[face] + tensor_permutation[grid_positions]

                interior_dofs = element.entity_dofs[topological_dimension][0]
                for cell in range(block.cell_count):
                    if interior_dofs:
                        local[
                            cell,
                            np.asarray(interior_dofs, dtype=np.int32),
                        ] = cell_starts[cell_offset + cell] + np.arange(
                            len(interior_dofs), dtype=np.int32
                        )
                if np.any(local < 0):
                    raise ValueError("High-order H1 entity map left unassigned DOFs.")
                orientation = np.ones_like(local, dtype=float)
            elif association == "vertex":
                if element.local_dof_count != vertices.shape[1]:
                    raise ValueError(
                        "Vertex-associated H1 elements require one DOF per vertex."
                    )
                local = vertices
                orientation = np.ones_like(local, dtype=float)
            else:
                raise ValueError("Unsupported finite-element DOF map.")
            block_dofs.append(jnp.asarray(local))
            orientations.append(jnp.asarray(orientation))
            relations.append(RowRelation(local, source_size=global_count))
            cell_offset += block.cell_count

        coordinate_weights = tuple(
            lagrange_element(block.cell_kind, 1).tabulate(element.reference_nodes)[0]
            for block, element in zip(mesh.blocks, resolved, strict=True)
        )
        if association == "cell":
            boundary = np.zeros((global_count,), dtype=bool)
            coordinate_blocks = []
            mesh_coordinates = np.asarray(mesh.coordinates)
            for block, weights_ in zip(mesh.blocks, coordinate_weights, strict=True):
                cell_coordinates = mesh_coordinates[
                    np.asarray(block.vertices, dtype=np.int32)
                ]
                mapped = oe.contract(
                    "ia,cad->cid",
                    np.asarray(weights_),
                    cell_coordinates,
                )
                coordinate_blocks.append(mapped.reshape((-1, mesh.ambient_dimension)))
            dof_coordinates = np.concatenate(tuple(coordinate_blocks), axis=0)
        elif association == "edge":
            if not isinstance(connectivity, PolygonalConnectivity):
                raise TypeError("Compatible edge map requires polygonal connectivity.")
            boundary = np.asarray(connectivity.boundary_edges, dtype=bool)
            edge_vertices = np.asarray(connectivity.edges, dtype=np.int32)
            dof_coordinates = np.mean(
                np.asarray(mesh.coordinates)[edge_vertices],
                axis=1,
            )
        else:
            boundary = np.zeros((global_count,), dtype=bool)
            boundary[:vertex_count] = np.asarray(
                mesh.topology.entity_sets[0].subset("boundary").mask,
                dtype=bool,
            )
            if association == "entity":
                if edge_starts is None or edge_widths is None:
                    raise TypeError("High-order H1 edge offsets are unavailable.")
                for edge in np.flatnonzero(
                    np.asarray(connectivity.boundary_edges, dtype=bool)
                ):
                    start = int(edge_starts[edge])
                    boundary[start : start + int(edge_widths[edge])] = True
                if isinstance(connectivity, HexahedralConnectivity):
                    if face_starts is None or face_shapes is None:
                        raise TypeError("High-order H1 face offsets are unavailable.")
                    for face in np.flatnonzero(
                        np.asarray(connectivity.boundary_faces, dtype=bool)
                    ):
                        start = int(face_starts[face])
                        boundary[
                            start : start
                            + prod(tuple(int(value) for value in face_shapes[face]))
                        ] = True
                accumulated = np.zeros(
                    (global_count, mesh.ambient_dimension),
                    dtype=np.asarray(mesh.coordinates).dtype,
                )
                counts = np.zeros((global_count,), dtype=np.int32)
                for block, weights_, routes in zip(
                    mesh.blocks,
                    coordinate_weights,
                    block_dofs,
                    strict=True,
                ):
                    mapped = oe.contract(
                        "ia,cad->cid",
                        np.asarray(weights_),
                        np.asarray(mesh.coordinates)[np.asarray(block.vertices)],
                    )
                    routes_ = np.asarray(routes)
                    np.add.at(
                        accumulated,
                        routes_.reshape((-1,)),
                        mapped.reshape((-1, mesh.ambient_dimension)),
                    )
                    np.add.at(counts, routes_.reshape((-1,)), 1)
                if np.any(counts == 0):
                    raise ValueError("High-order H1 coordinates contain unassigned DOFs.")
                dof_coordinates = accumulated / counts[:, None]
            else:
                dof_coordinates = np.asarray(mesh.coordinates)

        canonical_routes = []
        canonical_orientations = []
        for block, routes, orientation in zip(
            mesh.blocks,
            block_dofs,
            orientations,
            strict=True,
        ):
            order = np.argsort(np.asarray(block.global_ids), kind="stable")
            canonical_routes.append(np.asarray(routes)[order])
            canonical_orientations.append(np.asarray(orientation)[order])
        self.block_names = tuple(block.name for block in mesh.blocks)
        self.cell_dofs = tuple(block_dofs)
        self.orientations = tuple(orientations)
        self.cell_coordinate_weights = tuple(
            jnp.asarray(value) for value in coordinate_weights
        )
        self.relations = tuple(relations)
        self.global_dof_count = global_count
        self.entity_dof_counts = entity_dof_counts
        self.entity_dofs_per_entity = entity_dofs_per_entity
        self.component_shape = components
        self.association = association
        self.boundary_dof_mask = jnp.asarray(boundary)
        self.dof_coordinates = jnp.asarray(dof_coordinates)
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
                "cell_coordinate_weights": [
                    array_tree_fingerprint(np.asarray(value))
                    for value in coordinate_weights
                ],
            }
        )

    def evaluate_coordinates(
        self,
        mesh: CellMesh,
        coordinates: ArrayLike,
        /,
    ) -> Array:
        points = jnp.asarray(coordinates)
        if points.shape != mesh.coordinates.shape:
            raise ValueError("DOF coordinate evaluation must preserve mesh shape.")
        if self.association == "vertex":
            return points
        connectivity = mesh.connectivity
        if self.association == "edge":
            if not isinstance(connectivity, PolygonalConnectivity):
                raise TypeError("Edge DOF coordinates require polygonal connectivity.")
            edges = jnp.asarray(connectivity.edges, dtype=jnp.int32)
            return 0.5 * (points[edges[:, 0]] + points[edges[:, 1]])
        if self.association == "entity":
            accumulated = jnp.zeros(
                (self.global_dof_count, mesh.ambient_dimension),
                dtype=points.dtype,
            )
            counts = jnp.zeros((self.global_dof_count,), dtype=jnp.int32)
            for block, weights_, routes in zip(
                mesh.blocks,
                self.cell_coordinate_weights,
                self.cell_dofs,
                strict=True,
            ):
                mapped = oe.contract(
                    "ia,cad->cid",
                    weights_,
                    points[block.vertices],
                )
                accumulated = accumulated.at[routes].add(mapped)
                counts = counts.at[routes].add(1)
            return accumulated / counts[:, None]
        if self.association == "cell":
            coordinate_blocks = []
            for block, weights_ in zip(
                mesh.blocks,
                self.cell_coordinate_weights,
                strict=True,
            ):
                mapped = oe.contract(
                    "ia,cad->cid",
                    weights_,
                    points[block.vertices],
                )
                coordinate_blocks.append(mapped.reshape((-1, mesh.ambient_dimension)))
            return jnp.concatenate(tuple(coordinate_blocks), axis=0)
        raise ValueError("Unknown finite-element DOF association.")


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


class FiniteElementRuntimeData(StrictModule, NonTrainableState):
    """Dynamic fixed-topology geometry realization for FE execution."""

    coordinates: Array
    topology_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)
    numeric_version: str = eqx.field(static=True)
    runtime_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh: CellMesh,
        coordinates: ArrayLike,
        /,
        *,
        numeric_version: str,
        geometry_layout_id: str | None = None,
    ):
        if not isinstance(mesh, CellMesh):
            raise TypeError("mesh must be a CellMesh.")
        points = jnp.asarray(coordinates)
        if points.ndim != 2 or points.shape[1] != mesh.ambient_dimension:
            raise ValueError(
                "Finite-element runtime coordinates must preserve ambient dimension."
            )
        version = str(numeric_version)
        if not version:
            raise ValueError("numeric_version must be non-empty.")
        self.coordinates = points
        self.topology_id = mesh.topology_id
        layout_id = (
            mesh.geometry_layout_id
            if geometry_layout_id is None
            else str(geometry_layout_id)
        )
        if not layout_id:
            raise ValueError("geometry_layout_id must be non-empty.")
        self.geometry_layout_id = layout_id
        self.numeric_version = version
        self.runtime_id = canonical_fingerprint(
            {
                "kind": "finite-element-runtime",
                "topology": mesh.topology_id,
                "geometry_layout": layout_id,
                "numeric_version": version,
            }
        )


def _facet_routes(mesh: CellMesh, /) -> tuple[np.ndarray, ...]:
    connectivity = mesh.connectivity
    if isinstance(connectivity, IntervalConnectivity):
        cell_facets = np.asarray(connectivity.cell_vertices, dtype=np.int32)
        valid = np.ones_like(cell_facets, dtype=bool)
        facet_count = connectivity.vertex_count
    elif isinstance(connectivity, PolygonalConnectivity):
        cell_facets = np.asarray(connectivity.cell_edges, dtype=np.int32)
        valid = np.asarray(connectivity.cell_edge_valid, dtype=bool)
        facet_count = int(connectivity.edges.shape[0])
    elif isinstance(connectivity, PolyhedralConnectivity):
        cell_facets = np.asarray(connectivity.cell_faces, dtype=np.int32)
        valid = np.asarray(connectivity.cell_face_valid, dtype=bool)
        facet_count = int(connectivity.faces.shape[0])
    else:
        cell_facets = np.asarray(connectivity.cell_faces, dtype=np.int32)
        valid = np.ones_like(cell_facets, dtype=bool)
        facet_count = int(connectivity.faces.shape[0])
    owner = np.full((facet_count,), -1, dtype=np.int32)
    neighbour = np.full((facet_count,), -1, dtype=np.int32)
    owner_local = np.full((facet_count,), -1, dtype=np.int32)
    neighbour_local = np.full((facet_count,), -1, dtype=np.int32)
    for cell in range(cell_facets.shape[0]):
        for local in range(cell_facets.shape[1]):
            if not valid[cell, local]:
                continue
            facet = int(cell_facets[cell, local])
            if owner[facet] < 0:
                owner[facet] = cell
                owner_local[facet] = local
            else:
                neighbour[facet] = cell
                neighbour_local[facet] = local
    if np.any(owner < 0):
        raise ValueError("Every finite-element facet requires an owner cell.")
    return owner, neighbour, owner_local, neighbour_local


def _validate_mesh_geometry(mesh: CellMesh, /) -> None:
    coordinates = np.asarray(mesh.coordinates, dtype=float)
    for block in mesh.blocks:
        cells = np.asarray(block.vertices, dtype=np.int32)
        points = coordinates[cells]
        if block.cell_kind == "interval":
            difference = points[:, 1] - points[:, 0]
            determinant = np.sum(difference * difference, axis=-1)
        elif block.cell_kind == "triangle":
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
        elif block.cell_kind == "hexahedron":
            edge_matrix = np.stack(
                (
                    points[:, 1] - points[:, 0],
                    points[:, 3] - points[:, 0],
                    points[:, 4] - points[:, 0],
                ),
                axis=-1,
            )
            gram = np.swapaxes(edge_matrix, -1, -2) @ edge_matrix
            determinant = np.linalg.det(gram)
        elif block.cell_kind in ("prism", "pyramid"):
            tetrahedra = (
                ((0, 1, 2, 3), (1, 2, 4, 3), (2, 4, 5, 3))
                if block.cell_kind == "prism"
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
                gram = np.swapaxes(edge_matrix, -1, -2) @ edge_matrix
                determinants.append(np.linalg.det(gram))
            determinant = np.min(np.stack(tuple(determinants), axis=-1), axis=-1)
        else:
            raise ValueError("Unsupported finite-element cell kind.")
        if np.any(~np.isfinite(determinant)) or np.any(determinant <= 0.0):
            raise ValueError(
                "Finite-element cells require positive finite metric determinant."
            )


class FiniteElementPlan(AbstractDiscretizationPlan):
    mesh: CellMesh
    fields: tuple[FiniteElementFieldSpec, ...]
    coordinate_spec: FiniteElementCoordinateSpec
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
        coordinate_spec: FiniteElementCoordinateSpec | None = None,
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
        coordinates = (
            FiniteElementCoordinateSpec.affine(mesh)
            if coordinate_spec is None
            else coordinate_spec
        )
        if not isinstance(coordinates, FiniteElementCoordinateSpec):
            raise TypeError(
                "coordinate_spec must be FiniteElementCoordinateSpec or None."
            )
        coordinates.resolve(mesh)
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
        self.coordinate_spec = coordinates
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
                "coordinate_spec": coordinates.coordinate_spec_id,
                "precision_policy": precision.policy_id,
            }
        )

    def prepare(self, /, *, numeric_version: str = "0"):
        return FiniteElementDiscretization(self, numeric_version=numeric_version)


class FiniteElementDiscretization(AbstractPreparedLocalDiscretization):
    mesh: CellMesh
    dof_maps: tuple[FiniteElementDofMap, ...]
    default_runtime: FiniteElementRuntimeData
    elements: tuple[tuple[FiniteElementSpec, ...], ...]
    coordinate_elements: tuple[FiniteElementSpec, ...]
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
        coordinate_elements, coordinate_dofs, coordinate_values = (
            plan.coordinate_spec.resolve(mesh)
        )
        field_spaces = []
        dof_maps = []
        all_elements = []
        all_geometries = []
        cell_measures = None
        for field in plan.fields:
            elements = field.resolve(mesh)
            all_elements.append(elements)
            dof_map = FiniteElementDofMap(
                mesh,
                elements,
                component_shape=field.component_shape,
            )
            dof_maps.append(dof_map)
            vector_shape = (dof_map.global_dof_count,) + field.component_shape
            vector_space = ArraySpace(vector_shape)
            vertex_count = int(mesh.coordinates.shape[0])
            conformity = elements[0].conformity
            if dof_map.association == "vertex":
                layout = EntityDofLayout(
                    mesh.topology.entity_sets[0].entity_set_id,
                    vertex_count,
                    vertex_count,
                    component_shape=field.component_shape,
                )
            elif dof_map.association == "entity":
                entity_names = ("vertices", "edges", "faces", "cells")
                block_names = []
                layouts = []
                for dimension, (
                    entity_dof_count,
                    dofs_per_entity,
                ) in enumerate(
                    zip(
                        dof_map.entity_dof_counts,
                        dof_map.entity_dofs_per_entity,
                        strict=True,
                    )
                ):
                    if entity_dof_count == 0:
                        continue
                    entities = mesh.topology.entity_sets[dimension]
                    block_names.append(entity_names[dimension])
                    layouts.append(
                        EntityDofLayout(
                            entities.entity_set_id,
                            entities.count,
                            entity_dof_count,
                            dofs_per_entity=dofs_per_entity,
                            component_shape=field.component_shape,
                        )
                    )
                layout = BlockDofLayout(tuple(block_names), tuple(layouts))
            elif dof_map.association == "edge":
                edge_count = dof_map.global_dof_count
                layout = EntityDofLayout(
                    mesh.topology.entity_sets[1].entity_set_id,
                    edge_count,
                    edge_count,
                    component_shape=field.component_shape,
                )
            elif dof_map.association == "cell":
                cell_dof_count = dof_map.global_dof_count
                layout = EntityDofLayout(
                    mesh.topology.entity_sets[mesh.topological_dimension].entity_set_id,
                    cell_dof_count,
                    cell_dof_count,
                    component_shape=field.component_shape,
                )
            else:
                raise ValueError("Unknown finite-element DOF association.")
            representation = elements[0].representation
            field_spaces.append(
                DiscreteFieldSpace(
                    field.name,
                    mesh.support.support_id,
                    layout,
                    vector_space,
                    representation=representation,
                    conformity=conformity,
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
                _prepare_block_geometry(
                    mesh,
                    block,
                    element,
                    coordinates=coordinate_values,
                    coordinate_element=coordinate_element,
                    geometry_dofs=geometry_dofs,
                    precision_policy=plan.precision_policy,
                )
                for block, element, coordinate_element, geometry_dofs in zip(
                    mesh.blocks,
                    elements,
                    coordinate_elements,
                    coordinate_dofs,
                    strict=True,
                )
            )
            all_geometries.append(geometries)
            if cell_measures is None:
                cell_measures = jnp.concatenate(
                    tuple(geometry.measure for geometry in geometries),
                    axis=0,
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
        owner, neighbour, owner_local, neighbour_local = _facet_routes(mesh)
        exterior = np.flatnonzero(neighbour < 0)
        interior = np.flatnonzero(neighbour >= 0)
        cell_count = sum(block.cell_count for block in mesh.blocks)
        cell_entities = mesh.topology.entity_sets[mesh.topological_dimension]
        facet_entities = mesh.topology.entity_sets[mesh.topological_dimension - 1]
        self.mesh = mesh
        self.default_runtime = FiniteElementRuntimeData(
            mesh,
            coordinate_values,
            numeric_version=version,
            geometry_layout_id=plan.coordinate_spec.coordinate_spec_id,
        )
        self.dof_maps = tuple(dof_maps)
        self.elements = tuple(all_elements)
        self.coordinate_elements = coordinate_elements
        self.coordinate_dofs = coordinate_dofs
        self.block_geometries = tuple(all_geometries)
        self.cell_domain = IntegrationDomain(
            "cell",
            np.arange(cell_count, dtype=np.int32),
            mesh.support.support_id,
            cell_entities.entity_set_id,
            owner_cells=np.arange(cell_count, dtype=np.int32),
        )
        self.exterior_facet_domain = IntegrationDomain(
            "exterior_facet",
            exterior,
            mesh.support.support_id,
            facet_entities.entity_set_id,
            owner_cells=owner[exterior],
            neighbour_cells=neighbour[exterior],
            owner_local_entities=owner_local[exterior],
            neighbour_local_entities=neighbour_local[exterior],
        )
        self.interior_facet_domain = IntegrationDomain(
            "interior_facet",
            interior,
            mesh.support.support_id,
            facet_entities.entity_set_id,
            owner_cells=owner[interior],
            neighbour_cells=neighbour[interior],
            owner_local_entities=owner_local[interior],
            neighbour_local_entities=neighbour_local[interior],
        )
        self.key = plan.key
        self.precision_policy = plan.precision_policy
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
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-finite-element",
                "plan": plan.plan_id,
                "mesh": mesh.mesh_id,
                "numeric_version": version,
            }
        )

    @property
    def precision_evidence(self):
        return self.precision_policy.evidence()

    def prepare_runtime(
        self,
        coordinates: ArrayLike | None = None,
        /,
        *,
        numeric_version: str,
    ) -> FiniteElementRuntimeData:
        points = jnp.asarray(
            self.default_runtime.coordinates if coordinates is None else coordinates
        )
        if points.shape != self.default_runtime.coordinates.shape:
            raise ValueError(
                "Fixed-topology FE runtime coordinates must preserve coordinate shape."
            )
        return FiniteElementRuntimeData(
            self.mesh,
            points,
            numeric_version=numeric_version,
            geometry_layout_id=self.default_runtime.geometry_layout_id,
        )

    def local_variational_capabilities(self, /) -> LocalVariationalCapabilities:
        from ._local_provider import FiniteElementLocalProvider

        return FiniteElementLocalProvider(self).local_variational_capabilities()

    def local_field_binding(self, name: str, /) -> LocalFieldBinding:
        from ._local_provider import FiniteElementLocalProvider

        return FiniteElementLocalProvider(self).local_field_binding(name)

    def prepare_local_regions(
        self,
        domain: IntegrationDomain,
        /,
        *,
        field_names: tuple[str, ...],
        maximum_derivative_order: int,
        kernel_mode: str,
    ) -> tuple[PreparedLocalRegion, ...]:
        from ._local_provider import FiniteElementLocalProvider

        return FiniteElementLocalProvider(self).prepare_local_regions(
            domain,
            field_names=field_names,
            maximum_derivative_order=maximum_derivative_order,
            kernel_mode=kernel_mode,
        )

    def validate_local_runtime(self, runtime: object, /) -> None:
        from ._local_provider import FiniteElementLocalProvider

        FiniteElementLocalProvider(self).validate_local_runtime(runtime)

    @property
    def mass(self) -> SparseLinearMap:
        if len(self.field_spaces) != 1:
            raise ValueError("mass is only unambiguous for one field.")
        return self.assemble_field_operators(
            self.field_spaces[0].name,
            self.default_runtime,
        )[0]

    @property
    def stiffness(self) -> SparseLinearMap:
        if len(self.field_spaces) != 1:
            raise ValueError("stiffness is only unambiguous for one field.")
        return self.assemble_field_operators(
            self.field_spaces[0].name,
            self.default_runtime,
        )[1]

    @property
    def vertices(self) -> Array:
        return self.mesh.coordinates

    @property
    def boundary_dof_mask(self) -> Array:
        if len(self.dof_maps) != 1:
            raise ValueError("boundary_dof_mask is only unambiguous for one field.")
        return self.dof_maps[0].boundary_dof_mask

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
            raise TypeError("function must be callable.")
        field_index = self._field_index(field_name)
        realized = self.default_runtime if runtime is None else runtime
        conformity = self.elements[field_index][0].conformity
        if conformity in ("Hdiv", "Hcurl"):
            if not isinstance(self.mesh.connectivity, PolygonalConnectivity):
                raise ValueError(
                    "Compatible projection currently requires polygonal edges."
                )
            edge_vertices = jnp.asarray(self.mesh.connectivity.edges, dtype=jnp.int32)
            edge_points = realized.coordinates[edge_vertices]
            tangent = edge_points[:, 1] - edge_points[:, 0]
            midpoint = 0.5 * (edge_points[:, 0] + edge_points[:, 1])
            values = jnp.asarray(function(midpoint, args))
            if values.shape != tangent.shape:
                raise ValueError(
                    "Compatible edge projection function must return one vector "
                    "per edge midpoint."
                )
            if conformity == "Hcurl":
                moments = jnp.sum(values * tangent, axis=-1)
            else:
                normal_measure = jnp.stack((tangent[:, 1], -tangent[:, 0]), axis=-1)
                moments = jnp.sum(values * normal_measure, axis=-1)
            return self.field_spaces[field_index].vector_space.validate(moments)
        coordinates = self.dof_maps[field_index].evaluate_coordinates(
            self.mesh,
            realized.coordinates,
        )
        values = jnp.asarray(function(coordinates, args))
        return self.field_spaces[field_index].vector_space.validate(values)

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
            raise ValueError("Unknown finite-element integration-domain kind.")
        base = domains[kind]
        if selection is None:
            return base
        if not isinstance(selection, EntitySelection):
            raise TypeError("selection must be EntitySelection or None.")
        if selection.entity_set_id != base.entity_set_id:
            raise ValueError("Entity selection does not match the domain entity set.")
        entity_mask = np.asarray(selection.mask, dtype=bool)
        base_entities = np.asarray(base.entity_indices, dtype=np.int32)
        selected_rows = np.flatnonzero(entity_mask[base_entities])
        return IntegrationDomain(
            base.kind,
            base_entities[selected_rows],
            base.support_id,
            base.entity_set_id,
            owner_cells=np.asarray(base.owner_cells)[selected_rows],
            neighbour_cells=np.asarray(base.neighbour_cells)[selected_rows],
            owner_local_entities=np.asarray(base.owner_local_entities)[selected_rows],
            neighbour_local_entities=np.asarray(base.neighbour_local_entities)[
                selected_rows
            ],
            neighbour_trace_permutations=np.asarray(base.neighbour_trace_permutations)[
                selected_rows
            ],
            periodic_face_mask=np.asarray(base.periodic_face_mask)[selected_rows],
            selection_id=selection.selection_id,
        )

    def trace(
        self,
        field_name: str,
        coefficients: ArrayLike,
        /,
        *,
        runtime: FiniteElementRuntimeData | None = None,
    ) -> tuple[Array, Array]:
        field_index = self._field_index(field_name)
        values = self.field_spaces[field_index].vector_space.validate(coefficients)
        realized = self.default_runtime if runtime is None else runtime
        coordinates = self.dof_maps[field_index].evaluate_coordinates(
            self.mesh,
            realized.coordinates,
        )
        mask = self.dof_maps[field_index].boundary_dof_mask
        return coordinates[mask], values[mask]

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
        field_index = self._field_index(field_name)
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
        orientation = self.dof_maps[field_index].orientations[block_index]
        local = local * orientation.reshape(
            orientation.shape + (1,) * (local.ndim - orientation.ndim)
        )
        if geometry.basis_values.ndim == 2:
            return oe.contract("qi,ci...->cq...", geometry.basis_values, local)
        return oe.contract("cqiv,ci->cqv", geometry.basis_values, local)

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
        if points.shape != self.default_runtime.coordinates.shape:
            raise ValueError(
                "Fixed-topology FE geometry evaluation must preserve coordinate shape."
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
            )
            for block, element, coordinate_element, geometry_dofs in zip(
                self.mesh.blocks,
                self.elements[field_index],
                self.coordinate_elements,
                self.coordinate_dofs,
                strict=True,
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
        field_index = self._field_index(field_name)
        index = int(block_index)
        if index < 0 or index >= len(self.mesh.blocks):
            raise IndexError("block_index is outside the finite-element mesh.")
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
        )

    def assemble_field_operators(
        self,
        field_name: str,
        runtime: FiniteElementRuntimeData,
        /,
    ) -> tuple[SparseLinearMap, SparseLinearMap]:
        if not isinstance(runtime, FiniteElementRuntimeData):
            raise TypeError("runtime must be FiniteElementRuntimeData.")
        if (
            runtime.topology_id != self.mesh.topology_id
            or runtime.geometry_layout_id != self.default_runtime.geometry_layout_id
        ):
            raise ValueError("Finite-element runtime does not match this discretization.")
        field_index = self._field_index(field_name)
        geometries = self.evaluate_geometry(field_name, runtime.coordinates)
        mass_local = tuple(_local_mass_tensor(geometry) for geometry in geometries)
        stiffness_local = tuple(
            _local_stiffness_tensor(geometry) for geometry in geometries
        )
        dof_map = self.dof_maps[field_index]
        return (
            _assemble_local_operator(
                dof_map,
                mass_local,
                "finite-element-mass",
                positive_definite=True,
                component_shape=dof_map.component_shape,
            ),
            _assemble_local_operator(
                dof_map,
                stiffness_local,
                "finite-element-stiffness",
                positive_definite=False,
                component_shape=dof_map.component_shape,
            ),
        )


def _local_mass_tensor(geometry: FiniteElementBlockGeometry, /) -> Array:
    if geometry.basis_values.ndim == 2:
        return oe.contract(
            "cq,qi,qj->cij",
            geometry.physical_weights,
            geometry.basis_values,
            geometry.basis_values,
        )
    return oe.contract(
        "cq,cqiv,cqjv->cij",
        geometry.physical_weights,
        geometry.basis_values,
        geometry.basis_values,
    )


def _local_stiffness_tensor(geometry: FiniteElementBlockGeometry, /) -> Array:
    if geometry.physical_gradients.ndim == 4:
        return oe.contract(
            "cq,cqid,cqjd->cij",
            geometry.physical_weights,
            geometry.physical_gradients,
            geometry.physical_gradients,
        )
    return oe.contract(
        "cq,cqivd,cqjvd->cij",
        geometry.physical_weights,
        geometry.physical_gradients,
        geometry.physical_gradients,
    )


def _degree_aware_reference_rule(
    cell_kind: str, polynomial_degree: int, /
) -> tuple[Array, Array]:
    count = max(2, int(polynomial_degree) + 1)
    if cell_kind == "tetrahedron":
        count = max(count, int(polynomial_degree) + 2)
    axis, weights = np.polynomial.legendre.leggauss(count)
    axis = 0.5 * (axis + 1.0)
    weights = 0.5 * weights
    if cell_kind == "interval":
        return jnp.asarray(axis[:, None]), jnp.asarray(weights)
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
    if cell_kind == "prism":
        first, second, third = np.meshgrid(axis, axis, axis, indexing="ij")
        points = np.stack((first, (1.0 - first) * second, third), axis=-1)
        combined = (
            weights[:, None, None]
            * weights[None, :, None]
            * weights[None, None, :]
            * (1.0 - first)
        )
        return jnp.asarray(points.reshape((-1, 3))), jnp.asarray(combined.reshape((-1,)))
    if cell_kind == "pyramid":
        first, second, height = np.meshgrid(axis, axis, axis, indexing="ij")
        scale = 1.0 - height
        points = np.stack(
            (
                scale * first + 0.5 * height,
                scale * second + 0.5 * height,
                height,
            ),
            axis=-1,
        )
        combined = (
            weights[:, None, None]
            * weights[None, :, None]
            * weights[None, None, :]
            * scale**2
        )
        return jnp.asarray(points.reshape((-1, 3))), jnp.asarray(combined.reshape((-1,)))
    if cell_kind == "hexahedron":
        first, second, third = np.meshgrid(axis, axis, axis, indexing="ij")
        points = np.stack((first, second, third), axis=-1)
        combined = (
            weights[:, None, None] * weights[None, :, None] * weights[None, None, :]
        )
        return jnp.asarray(points.reshape((-1, 3))), jnp.asarray(combined.reshape((-1,)))
    raise ValueError("Unsupported finite-element cell kind.")


def _prepare_block_geometry(
    mesh: CellMesh,
    block: CellBlock,
    element: FiniteElementSpec,
    /,
    *,
    coordinates: ArrayLike | None = None,
    reference_points: ArrayLike | None = None,
    reference_weights: ArrayLike | None = None,
    coordinate_element: FiniteElementSpec | None = None,
    geometry_dofs: ArrayLike | None = None,
    precision_policy: FiniteElementPrecisionPolicy,
) -> FiniteElementBlockGeometry:
    if (reference_points is None) != (reference_weights is None):
        raise ValueError(
            "reference_points and reference_weights must be supplied together."
        )
    geometry_element = (
        lagrange_element(block.cell_kind, 1)
        if coordinate_element is None
        else coordinate_element
    )
    if reference_points is None:
        points_, weights_ = _degree_aware_reference_rule(
            block.cell_kind,
            max(element.degree, geometry_element.degree),
        )
    else:
        points_ = jnp.asarray(reference_points)
        weights_ = jnp.asarray(reference_weights)
    reference_points = precision_policy.geometry(points_)
    reference_weights = precision_policy.accumulation(weights_)
    geometry_values, geometry_gradients = geometry_element.tabulate(reference_points)
    basis_values, reference_gradients = element.tabulate(reference_points)
    geometry_values = precision_policy.geometry(geometry_values)
    geometry_gradients = precision_policy.geometry(geometry_gradients)
    basis_values = precision_policy.evaluation(basis_values)
    reference_gradients = precision_policy.evaluation(reference_gradients)
    coordinate_values = precision_policy.geometry(
        mesh.coordinates if coordinates is None else coordinates
    )
    coordinate_routes = (
        block.vertices if geometry_dofs is None else jnp.asarray(geometry_dofs)
    )
    cell_coordinates = coordinate_values[coordinate_routes]
    physical_points = oe.contract("qi,cid->cqd", geometry_values, cell_coordinates)
    jacobian = oe.contract("qir,cid->cqdr", geometry_gradients, cell_coordinates)
    metric = oe.contract("cqdi,cqdj->cqij", jacobian, jacobian)
    inverse_result = inverse_small_linear(
        SmallLinearSolvePlan(metric.shape[-1]),
        metric,
    )
    inverse_metric = inverse_result.value
    determinant = inverse_result.determinant
    measure_factor = jnp.sqrt(determinant)
    measure_factor = eqx.error_if(
        measure_factor,
        jnp.any(
            ~inverse_result.successful
            | ~jnp.isfinite(measure_factor)
            | (measure_factor <= 0.0)
        ),
        "Finite-element geometry requires positive finite metric determinant.",
    )
    inverse_jacobian = oe.contract(
        "cqij,cqdj->cqid",
        inverse_metric,
        jacobian,
    )
    if element.mapping == "identity":
        physical_basis = basis_values
        physical_gradients = oe.contract(
            "cqdi,cqij,qkj->cqkd",
            jacobian,
            inverse_metric,
            reference_gradients,
        )
    elif element.mapping == "contravariant_piola":
        physical_basis = (
            oe.contract(
                "cqdm,qkm->cqkd",
                jacobian,
                basis_values,
            )
            / measure_factor[..., None, None]
        )
        physical_gradients = (
            oe.contract(
                "cqdm,qkmr,cqrn->cqkdn",
                jacobian,
                reference_gradients,
                inverse_jacobian,
            )
            / measure_factor[..., None, None, None]
        )
    elif element.mapping == "covariant_piola":
        physical_basis = oe.contract(
            "cqmd,qkm->cqkd",
            inverse_jacobian,
            basis_values,
        )
        physical_gradients = oe.contract(
            "cqmd,qkmr,cqrn->cqkdn",
            inverse_jacobian,
            reference_gradients,
            inverse_jacobian,
        )
    else:
        raise ValueError(f"Unsupported finite-element mapping {element.mapping!r}.")
    physical_weights = precision_policy.accumulation(
        measure_factor * reference_weights[None, :]
    )
    return FiniteElementBlockGeometry(
        block_name=block.name,
        reference_points=reference_points,
        reference_weights=reference_weights,
        basis_values=physical_basis,
        reference_gradients=reference_gradients,
        physical_points=physical_points,
        physical_gradients=physical_gradients,
        physical_weights=physical_weights,
        measure=precision_policy.output(jnp.sum(physical_weights, axis=1)),
    )


def _assemble_local_operator(
    dof_map: FiniteElementDofMap,
    local_values: Sequence[Array],
    kind: str,
    /,
    *,
    positive_definite: bool,
    component_shape: Sequence[int] = (),
) -> SparseLinearMap:
    source_parts = []
    target_parts = []
    coefficient_parts = []
    component_count = (
        prod(tuple(int(size) for size in component_shape)) if component_shape else 1
    )
    for cell_dofs, values in zip(dof_map.cell_dofs, local_values, strict=True):
        indices = np.asarray(cell_dofs, dtype=np.int32)
        width = indices.shape[1]
        components = np.arange(component_count, dtype=np.int32)
        flat = indices[..., None] * component_count + components
        source_parts.append(
            np.broadcast_to(
                flat[:, None, :, :],
                (indices.shape[0], width, width, component_count),
            ).reshape((-1,))
        )
        target_parts.append(
            np.broadcast_to(
                flat[:, :, None, :],
                (indices.shape[0], width, width, component_count),
            ).reshape((-1,))
        )
        coefficient_parts.append(
            jnp.broadcast_to(
                jnp.asarray(values)[..., None],
                values.shape + (component_count,),
            ).reshape((-1,))
        )
    relation = EdgeRelation(
        np.concatenate(source_parts),
        np.concatenate(target_parts),
        source_size=dof_map.global_dof_count * component_count,
        target_size=dof_map.global_dof_count * component_count,
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
    "FiniteElementCoordinateSpec",
    "FiniteElementDiscretization",
    "FiniteElementDofMap",
    "FiniteElementFieldSpec",
    "FiniteElementPlan",
    "FiniteElementRuntimeData",
    "IntegrationDomain",
]
