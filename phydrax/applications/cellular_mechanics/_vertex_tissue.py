#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-capacity polygonal and polyhedral vertex-tissue mechanics.

The compiled mechanics path is deliberately separated from host-side topology
transactions. A prepared epoch owns immutable incidence arrays and stable entity
identities; topology changes produce a candidate epoch, explicit evidence, and an
atomic commit or rollback.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


VertexTissueDimension: TypeAlias = Literal[2, 3]


class VertexTissueStatus(IntEnum):
    """Fail-closed status shared by mechanics, dynamics, and topology epochs."""

    SUCCESS = 0
    NONFINITE = 1
    ORIENTATION_FAILURE = 2
    QUALITY_FAILURE = 3
    NONMANIFOLD = 4
    CAPACITY_EXCEEDED = 5
    STALE_EPOCH = 6
    INVALID_EVENT = 7
    CONSERVATION_FAILURE = 8
    LINEAGE_FAILURE = 9
    ROLLED_BACK = 10


class VertexTissueEventKind(IntEnum):
    """Discrete transitions supported at accepted host-side epoch boundaries."""

    T1 = 0
    T2 = 1
    T3 = 2
    DIVISION = 3
    EXTRUSION = 4
    APOPTOSIS = 5
    FACE_TRANSITION = 6
    EDGE_TRANSITION = 7


def _real_array(name: str, value: ArrayLike, rank: int, /) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}.")
    if not np.issubdtype(array.dtype, np.inexact) or np.iscomplexobj(array):
        raise TypeError(f"{name} must be a real inexact array.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _integer_array(name: str, value: ArrayLike, rank: int, /) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}.")
    if not np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.bool_):
        raise TypeError(f"{name} must be an integer array.")
    limits = np.iinfo(np.int32)
    if np.any(array < limits.min) or np.any(array > limits.max):
        raise ValueError(f"{name} values must fit in int32.")
    return array.astype(np.int32, copy=False)


def _identifier_array(name: str, value: ArrayLike, /) -> tuple[np.ndarray, np.ndarray]:
    identifiers = _integer_array(name, value, 1)
    if identifiers.size == 0:
        raise ValueError(f"{name} must have positive capacity.")
    if np.any(identifiers < -1):
        raise ValueError(f"{name} uses -1 as its only inactive identifier.")
    active = identifiers >= 0
    active_ids = identifiers[active]
    if np.unique(active_ids).size != active_ids.size:
        raise ValueError(f"Active {name} must be unique.")
    return identifiers, active


def _parameter(
    name: str,
    value: ArrayLike,
    count: int,
    /,
    *,
    nonnegative: bool = False,
    positive_active: np.ndarray | None = None,
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape == ():
        array = np.full((count,), array, dtype=np.result_type(array, np.float32))
    array = _real_array(name, array, 1)
    if array.shape != (count,):
        raise ValueError(f"{name} must be scalar or have shape ({count},).")
    if nonnegative and np.any(array < 0.0):
        raise ValueError(f"{name} must be nonnegative.")
    if positive_active is not None and np.any(array[positive_active] <= 0.0):
        raise ValueError(f"Active {name} values must be positive.")
    return array


def _compact_row(row: np.ndarray, name: str, /) -> np.ndarray:
    active = row >= 0
    if np.any(row < -1):
        raise ValueError(f"{name} uses -1 as its only padding value.")
    if np.any(active[1:] & ~active[:-1]):
        raise ValueError(f"{name} padding must be trailing.")
    return row[active]


def _validate_edge_rows(
    edge_vertices: np.ndarray,
    edge_active: np.ndarray,
    vertex_active: np.ndarray,
    /,
) -> None:
    edge_count = edge_active.size
    vertex_count = vertex_active.size
    if edge_vertices.shape != (edge_count, 2):
        raise ValueError(f"edge_vertex_indices must have shape ({edge_count}, 2).")
    if np.any(edge_vertices[~edge_active] != -1):
        raise ValueError("Inactive edge incidence rows must be entirely -1.")
    used_vertices: set[int] = set()
    undirected_edges: set[tuple[int, int]] = set()
    for edge in np.flatnonzero(edge_active):
        endpoints = edge_vertices[edge]
        if (
            np.any(endpoints < 0)
            or np.any(endpoints >= vertex_count)
            or endpoints[0] == endpoints[1]
            or not np.all(vertex_active[endpoints])
        ):
            raise ValueError("Active edges must join two distinct active vertices.")
        undirected = tuple(sorted((int(endpoints[0]), int(endpoints[1]))))
        if undirected in undirected_edges:
            raise ValueError("Active edges must represent distinct undirected edges.")
        undirected_edges.add(undirected)
        used_vertices.update((int(endpoints[0]), int(endpoints[1])))
    if used_vertices != set(int(index) for index in np.flatnonzero(vertex_active)):
        raise ValueError("Every active vertex must be incident to an active edge.")


def _interface_row_cells(row: np.ndarray, cell_active: np.ndarray, /) -> tuple[int, ...]:
    if row.shape != (2,) or np.any(row < -1):
        raise ValueError("Interface cell rows must contain two indices or -1 padding.")
    cells = tuple(int(value) for value in row if value >= 0)
    if len(cells) not in (1, 2) or len(set(cells)) != len(cells):
        raise ValueError("An active interface must have one or two distinct cells.")
    if any(value >= cell_active.size or not cell_active[value] for value in cells):
        raise ValueError("Interface incidence refers to an inactive or absent cell.")
    return tuple(sorted(cells))


def _validate_polygonal_topology(
    edge_vertices: np.ndarray,
    edge_active: np.ndarray,
    cell_edges: np.ndarray,
    cell_edge_orientations: np.ndarray,
    cell_active: np.ndarray,
    interface_cells: np.ndarray,
    /,
) -> None:
    cell_count = cell_active.size
    edge_count = edge_active.size
    if cell_edges.ndim != 2 or cell_edges.shape[0] != cell_count:
        raise ValueError("cell_edge_indices must have shape (cell_capacity, width).")
    if cell_edge_orientations.shape != cell_edges.shape:
        raise ValueError("cell_edge_orientations must match cell_edge_indices.")
    if interface_cells.shape != (edge_count, 2):
        raise ValueError(f"interface_cell_indices must have shape ({edge_count}, 2).")
    incident: list[list[int]] = [[] for _ in range(edge_count)]
    incidence_orientation: list[list[int]] = [[] for _ in range(edge_count)]
    for cell in range(cell_count):
        edges = _compact_row(cell_edges[cell], "cell_edge_indices")
        orientations = cell_edge_orientations[cell]
        valid = cell_edges[cell] >= 0
        if (
            np.any(orientations[valid] == 0)
            or np.any(np.abs(orientations[valid]) != 1)
            or np.any(orientations[~valid] != 0)
        ):
            raise ValueError(
                "Cell-edge orientations must be ±1 on incidence and 0 on padding."
            )
        if not cell_active[cell]:
            if edges.size or np.any(orientations != 0):
                raise ValueError("Inactive cell incidence rows must be empty.")
            continue
        if edges.size < 3 or np.unique(edges).size != edges.size:
            raise ValueError(
                "Active polygonal cells require at least three distinct edges."
            )
        if np.any(edges >= edge_count) or not np.all(edge_active[edges]):
            raise ValueError("Cell-edge incidence refers to an inactive or absent edge.")
        starts: list[int] = []
        ends: list[int] = []
        for slot, edge in enumerate(edges):
            endpoints = edge_vertices[edge]
            if orientations[slot] > 0:
                start, end = int(endpoints[0]), int(endpoints[1])
            else:
                start, end = int(endpoints[1]), int(endpoints[0])
            starts.append(start)
            ends.append(end)
            incident[int(edge)].append(cell)
            incidence_orientation[int(edge)].append(int(orientations[slot]))
        if any(
            ends[index] != starts[(index + 1) % edges.size] for index in range(edges.size)
        ):
            raise ValueError("Oriented cell edges must form one closed polygonal loop.")
    for edge in range(edge_count):
        if not edge_active[edge]:
            if np.any(interface_cells[edge] != -1):
                raise ValueError("Inactive interface incidence rows must be entirely -1.")
            continue
        expected = tuple(sorted(incident[edge]))
        if len(expected) not in (1, 2):
            raise ValueError("Each active polygonal edge must bound one or two cells.")
        if len(expected) == 2 and sum(incidence_orientation[edge]) != 0:
            raise ValueError(
                "Adjacent polygonal cells must traverse their shared edge oppositely."
            )
        if _interface_row_cells(interface_cells[edge], cell_active) != expected:
            raise ValueError("Edge-cell and cell-edge incidence do not agree.")


def _face_edges(vertices: np.ndarray, /) -> tuple[tuple[int, int], ...]:
    return tuple(
        (int(vertices[index]), int(vertices[(index + 1) % vertices.size]))
        for index in range(vertices.size)
    )


def _canonical_cycle(vertices: np.ndarray, /) -> tuple[int, ...]:
    values = tuple(int(value) for value in vertices)
    rotations = tuple(values[index:] + values[:index] for index in range(len(values)))
    reversed_values = tuple(reversed(values))
    reverse_rotations = tuple(
        reversed_values[index:] + reversed_values[:index]
        for index in range(len(reversed_values))
    )
    return min(rotations + reverse_rotations)


def _validate_polyhedral_topology(
    edge_vertices: np.ndarray,
    edge_active: np.ndarray,
    vertex_active: np.ndarray,
    face_active: np.ndarray,
    face_vertices: np.ndarray,
    cell_active: np.ndarray,
    cell_faces: np.ndarray,
    cell_face_orientations: np.ndarray,
    interface_cells: np.ndarray,
    /,
) -> None:
    face_count = face_active.size
    cell_count = cell_active.size
    vertex_count = vertex_active.size
    if face_vertices.ndim != 2 or face_vertices.shape[0] != face_count:
        raise ValueError("face_vertex_indices must have shape (face_capacity, width).")
    if cell_faces.ndim != 2 or cell_faces.shape[0] != cell_count:
        raise ValueError("cell_face_indices must have shape (cell_capacity, width).")
    if cell_face_orientations.shape != cell_faces.shape:
        raise ValueError("cell_face_orientations must match cell_face_indices.")
    if interface_cells.shape != (face_count, 2):
        raise ValueError(f"interface_cell_indices must have shape ({face_count}, 2).")

    declared_edges = {
        tuple(sorted((int(row[0]), int(row[1])))) for row in edge_vertices[edge_active]
    }
    if len(declared_edges) != int(np.sum(edge_active)):
        raise ValueError("Active edges must represent distinct undirected edges.")
    used_edges: set[tuple[int, int]] = set()
    face_rows: list[np.ndarray] = []
    canonical_faces: set[tuple[int, ...]] = set()
    for face in range(face_count):
        vertices = _compact_row(face_vertices[face], "face_vertex_indices")
        face_rows.append(vertices)
        if not face_active[face]:
            if vertices.size or np.any(interface_cells[face] != -1):
                raise ValueError("Inactive face incidence rows must be empty.")
            continue
        if (
            vertices.size < 3
            or np.unique(vertices).size != vertices.size
            or np.any(vertices >= vertex_count)
            or not np.all(vertex_active[vertices])
        ):
            raise ValueError(
                "Active faces require at least three distinct active vertices."
            )
        canonical = _canonical_cycle(vertices)
        if canonical in canonical_faces:
            raise ValueError("Active faces must represent distinct polygonal interfaces.")
        canonical_faces.add(canonical)
        for directed in _face_edges(vertices):
            edge = tuple(sorted(directed))
            if edge not in declared_edges:
                raise ValueError(
                    "Every face edge must be present in edge_vertex_indices."
                )
            used_edges.add(edge)
    if used_edges != declared_edges:
        raise ValueError("Every active edge must occur on at least one active face.")

    incident: list[list[int]] = [[] for _ in range(face_count)]
    incidence_orientation: list[list[int]] = [[] for _ in range(face_count)]
    for cell in range(cell_count):
        faces = _compact_row(cell_faces[cell], "cell_face_indices")
        orientations = cell_face_orientations[cell]
        valid = cell_faces[cell] >= 0
        if np.any(np.abs(orientations[valid]) != 1) or np.any(orientations[~valid] != 0):
            raise ValueError(
                "Cell-face orientations must be ±1 on incidence and 0 on padding."
            )
        if not cell_active[cell]:
            if faces.size or np.any(orientations != 0):
                raise ValueError("Inactive cell incidence rows must be empty.")
            continue
        if faces.size < 4 or np.unique(faces).size != faces.size:
            raise ValueError(
                "Active polyhedral cells require at least four distinct faces."
            )
        if np.any(faces >= face_count) or not np.all(face_active[faces]):
            raise ValueError("Cell-face incidence refers to an inactive or absent face.")
        balance: dict[tuple[int, int], int] = {}
        occurrences: dict[tuple[int, int], int] = {}
        for slot, face in enumerate(faces):
            orientation = int(orientations[slot])
            vertices = face_rows[int(face)]
            if orientation < 0:
                vertices = vertices[::-1]
            for start, end in _face_edges(vertices):
                undirected = tuple(sorted((start, end)))
                sign = 1 if start < end else -1
                balance[undirected] = balance.get(undirected, 0) + sign
                occurrences[undirected] = occurrences.get(undirected, 0) + 1
            incident[int(face)].append(cell)
            incidence_orientation[int(face)].append(orientation)
        if any(value != 0 for value in balance.values()) or any(
            value != 2 for value in occurrences.values()
        ):
            raise ValueError(
                "Oriented cell faces must form one closed two-manifold boundary."
            )

    for face in range(face_count):
        if not face_active[face]:
            continue
        expected = tuple(sorted(incident[face]))
        if len(expected) not in (1, 2):
            raise ValueError("Each active face must bound one or two cells.")
        if _interface_row_cells(interface_cells[face], cell_active) != expected:
            raise ValueError("Face-cell and cell-face incidence do not agree.")
        if len(expected) == 2 and sum(incidence_orientation[face]) != 0:
            raise ValueError(
                "Interior face orientations must be opposite in adjacent cells."
            )


def _enum_kind(value: VertexTissueEventKind | int, /) -> VertexTissueEventKind:
    if isinstance(value, VertexTissueEventKind):
        return value
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError("kind must be a VertexTissueEventKind.")
    integer = int(value)
    for member in VertexTissueEventKind:
        if member.value == integer:
            return member
    raise ValueError("kind is not a supported vertex-tissue event.")


class VertexTissuePlan(StrictModule, NonTrainableState):
    """Immutable fixed-capacity tissue topology and constitutive parameters.

    Entity identifiers are stable nonnegative integers. A value of ``-1`` marks
    an inactive capacity slot. Incidence arrays contain local capacity indices,
    not stable IDs, and use trailing ``-1`` padding.
    """

    dimension: int = eqx.field(static=True)
    vertex_ids: Array
    edge_ids: Array
    face_ids: Array
    cell_ids: Array
    vertex_active: Array
    edge_active: Array
    face_active: Array
    cell_active: Array
    edge_vertex_indices: Array
    face_vertex_indices: Array
    cell_edge_indices: Array
    cell_edge_orientations: Array
    cell_face_indices: Array
    cell_face_orientations: Array
    interface_cell_indices: Array
    target_cell_measure: Array
    cell_measure_stiffness: Array
    target_boundary_measure: Array
    boundary_stiffness: Array
    interface_tension: Array
    cell_types: Array
    adhesion_matrix: Array
    active_contractility: Array
    cell_traction: Array
    vertex_drag: Array
    cell_parent_ids: Array
    cell_generation: Array
    field_names: tuple[str, ...] = eqx.field(static=True)
    minimum_edge_length: float = eqx.field(static=True)
    minimum_cell_measure: float = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        dimension: VertexTissueDimension,
        vertex_ids: ArrayLike,
        edge_ids: ArrayLike,
        edge_vertex_indices: ArrayLike,
        cell_ids: ArrayLike,
        interface_cell_indices: ArrayLike,
        target_cell_measure: ArrayLike,
        cell_measure_stiffness: ArrayLike,
        target_boundary_measure: ArrayLike,
        boundary_stiffness: ArrayLike,
        /,
        *,
        face_ids: ArrayLike | None = None,
        face_vertex_indices: ArrayLike | None = None,
        cell_edge_indices: ArrayLike | None = None,
        cell_edge_orientations: ArrayLike | None = None,
        cell_face_indices: ArrayLike | None = None,
        cell_face_orientations: ArrayLike | None = None,
        interface_tension: ArrayLike = 0.0,
        cell_types: ArrayLike | None = None,
        adhesion_matrix: ArrayLike | None = None,
        active_contractility: ArrayLike = 0.0,
        cell_traction: ArrayLike | None = None,
        vertex_drag: ArrayLike = 1.0,
        cell_parent_ids: ArrayLike | None = None,
        cell_generation: ArrayLike | None = None,
        field_names: tuple[str, ...] = (),
        minimum_edge_length: float = 1.0e-8,
        minimum_cell_measure: float = 1.0e-10,
    ):
        if isinstance(dimension, bool) or dimension not in (2, 3):
            raise ValueError("dimension must be 2 or 3.")
        dimension_ = int(dimension)
        vertices, vertex_active = _identifier_array("vertex_ids", vertex_ids)
        edges, edge_active = _identifier_array("edge_ids", edge_ids)
        cells, cell_active = _identifier_array("cell_ids", cell_ids)
        if not np.any(cell_active):
            raise ValueError("At least one cell capacity slot must be active.")
        if not np.any(vertex_active) or not np.any(edge_active):
            raise ValueError("A tissue requires active vertices and edges.")
        edge_vertices = _integer_array("edge_vertex_indices", edge_vertex_indices, 2)
        _validate_edge_rows(edge_vertices, edge_active, vertex_active)
        cell_count = cells.size
        edge_count = edges.size

        if dimension_ == 2:
            if face_ids is not None or face_vertex_indices is not None:
                raise ValueError("Polygonal tissue does not accept face incidence.")
            if cell_edge_indices is None or cell_edge_orientations is None:
                raise ValueError(
                    "Polygonal tissue requires explicit cell-edge incidence."
                )
            if cell_face_indices is not None or cell_face_orientations is not None:
                raise ValueError("Polygonal tissue does not accept cell-face incidence.")
            faces = np.empty((0,), dtype=np.int32)
            face_active = np.empty((0,), dtype=bool)
            face_vertices = np.empty((0, 0), dtype=np.int32)
            cell_edges = _integer_array("cell_edge_indices", cell_edge_indices, 2)
            edge_orientations = _integer_array(
                "cell_edge_orientations", cell_edge_orientations, 2
            )
            cell_faces = np.empty((cell_count, 0), dtype=np.int32)
            face_orientations = np.empty((cell_count, 0), dtype=np.int32)
            interface_count = edge_count
            interfaces = _integer_array(
                "interface_cell_indices", interface_cell_indices, 2
            )
            _validate_polygonal_topology(
                edge_vertices,
                edge_active,
                cell_edges,
                edge_orientations,
                cell_active,
                interfaces,
            )
        else:
            if face_ids is None or face_vertex_indices is None:
                raise ValueError("Polyhedral tissue requires explicit face incidence.")
            if cell_face_indices is None or cell_face_orientations is None:
                raise ValueError(
                    "Polyhedral tissue requires explicit cell-face incidence."
                )
            if cell_edge_indices is not None or cell_edge_orientations is not None:
                raise ValueError("Polyhedral tissue derives cell edges from its faces.")
            faces, face_active = _identifier_array("face_ids", face_ids)
            face_vertices = _integer_array("face_vertex_indices", face_vertex_indices, 2)
            cell_faces = _integer_array("cell_face_indices", cell_face_indices, 2)
            face_orientations = _integer_array(
                "cell_face_orientations", cell_face_orientations, 2
            )
            cell_edges = np.empty((cell_count, 0), dtype=np.int32)
            edge_orientations = np.empty((cell_count, 0), dtype=np.int32)
            interface_count = faces.size
            interfaces = _integer_array(
                "interface_cell_indices", interface_cell_indices, 2
            )
            _validate_polyhedral_topology(
                edge_vertices,
                edge_active,
                vertex_active,
                face_active,
                face_vertices,
                cell_active,
                cell_faces,
                face_orientations,
                interfaces,
            )

        target = _parameter(
            "target_cell_measure",
            target_cell_measure,
            cell_count,
            positive_active=cell_active,
        )
        measure_stiffness = _parameter(
            "cell_measure_stiffness",
            cell_measure_stiffness,
            cell_count,
            nonnegative=True,
        )
        boundary_target = _parameter(
            "target_boundary_measure",
            target_boundary_measure,
            cell_count,
            positive_active=cell_active,
        )
        boundary_modulus = _parameter(
            "boundary_stiffness", boundary_stiffness, cell_count, nonnegative=True
        )
        tension = _parameter(
            "interface_tension", interface_tension, interface_count, nonnegative=True
        )
        contractility = _parameter(
            "active_contractility", active_contractility, cell_count, nonnegative=True
        )
        drag = _parameter(
            "vertex_drag",
            vertex_drag,
            vertices.size,
            positive_active=vertex_active,
        )

        types = (
            np.where(cell_active, 0, -1).astype(np.int32)
            if cell_types is None
            else _integer_array("cell_types", cell_types, 1)
        )
        if (
            types.shape != (cell_count,)
            or np.any(types[~cell_active] != -1)
            or np.any(types[cell_active] < 0)
        ):
            raise ValueError(
                "cell_types must be nonnegative on active cells and -1 otherwise."
            )
        type_count = int(np.max(types[cell_active])) + 1
        adhesion = (
            np.zeros((type_count, type_count), dtype=target.dtype)
            if adhesion_matrix is None
            else _real_array("adhesion_matrix", adhesion_matrix, 2)
        )
        if adhesion.shape != (type_count, type_count):
            raise ValueError(
                f"adhesion_matrix must have shape ({type_count}, {type_count})."
            )
        tolerance = (
            100.0
            * np.finfo(adhesion.dtype).eps
            * max(1.0, float(np.max(np.abs(adhesion))))
        )
        if np.any(adhesion < 0.0) or not np.allclose(
            adhesion,
            adhesion.T,
            rtol=100.0 * np.finfo(adhesion.dtype).eps,
            atol=tolerance,
        ):
            raise ValueError("adhesion_matrix must be symmetric and nonnegative.")

        traction = (
            np.zeros((cell_count, dimension_), dtype=target.dtype)
            if cell_traction is None
            else _real_array("cell_traction", cell_traction, 2)
        )
        if traction.shape != (cell_count, dimension_):
            raise ValueError(
                f"cell_traction must have shape ({cell_count}, {dimension_})."
            )
        if np.any(traction[~cell_active] != 0.0):
            raise ValueError("Inactive cells must have zero traction.")

        parents = (
            np.where(cell_active, cells, -1).astype(np.int32)
            if cell_parent_ids is None
            else _integer_array("cell_parent_ids", cell_parent_ids, 1)
        )
        generations = (
            np.where(cell_active, 0, -1).astype(np.int32)
            if cell_generation is None
            else _integer_array("cell_generation", cell_generation, 1)
        )
        if (
            parents.shape != (cell_count,)
            or generations.shape != (cell_count,)
            or np.any(parents[~cell_active] != -1)
            or np.any(generations[~cell_active] != -1)
            or np.any(parents[cell_active] < 0)
            or np.any(generations[cell_active] < 0)
        ):
            raise ValueError(
                "Cell lineage must be nonnegative on active cells and -1 otherwise."
            )

        names = tuple(field_names)
        if any(
            not isinstance(name, str) or not name or name != name.strip()
            for name in names
        ):
            raise ValueError("field_names must contain nonempty canonical strings.")
        if len(set(names)) != len(names):
            raise ValueError("field_names must be unique.")
        minimum_edge = float(minimum_edge_length)
        minimum_measure = float(minimum_cell_measure)
        if (
            not np.isfinite(minimum_edge)
            or minimum_edge <= 0.0
            or not np.isfinite(minimum_measure)
            or minimum_measure <= 0.0
        ):
            raise ValueError("Geometry quality thresholds must be finite and positive.")

        topology_arrays = {
            "vertex_ids": vertices,
            "edge_ids": edges,
            "face_ids": faces,
            "cell_ids": cells,
            "edge_vertices": edge_vertices,
            "face_vertices": face_vertices,
            "cell_edges": cell_edges,
            "cell_edge_orientations": edge_orientations,
            "cell_faces": cell_faces,
            "cell_face_orientations": face_orientations,
            "interface_cells": interfaces,
            "cell_parent_ids": parents,
            "cell_generation": generations,
        }
        topology_id = canonical_fingerprint(
            {
                "kind": "fixed-capacity-vertex-tissue-topology",
                "dimension": dimension_,
                "incidence": array_tree_fingerprint(topology_arrays),
            }
        )
        plan_id = canonical_fingerprint(
            {
                "kind": "vertex-tissue-plan",
                "topology": topology_id,
                "field_names": list(names),
                "minimum_edge_length": minimum_edge,
                "minimum_cell_measure": minimum_measure,
                "constitutive": array_tree_fingerprint(
                    {
                        "target": target,
                        "measure_stiffness": measure_stiffness,
                        "boundary_target": boundary_target,
                        "boundary_stiffness": boundary_modulus,
                        "interface_tension": tension,
                        "cell_types": types,
                        "adhesion": adhesion,
                        "contractility": contractility,
                        "traction": traction,
                        "drag": drag,
                    }
                ),
            }
        )
        self.dimension = dimension_
        self.vertex_ids = jnp.asarray(vertices)
        self.edge_ids = jnp.asarray(edges)
        self.face_ids = jnp.asarray(faces)
        self.cell_ids = jnp.asarray(cells)
        self.vertex_active = jnp.asarray(vertex_active)
        self.edge_active = jnp.asarray(edge_active)
        self.face_active = jnp.asarray(face_active)
        self.cell_active = jnp.asarray(cell_active)
        self.edge_vertex_indices = jnp.asarray(edge_vertices)
        self.face_vertex_indices = jnp.asarray(face_vertices)
        self.cell_edge_indices = jnp.asarray(cell_edges)
        self.cell_edge_orientations = jnp.asarray(edge_orientations)
        self.cell_face_indices = jnp.asarray(cell_faces)
        self.cell_face_orientations = jnp.asarray(face_orientations)
        self.interface_cell_indices = jnp.asarray(interfaces)
        self.target_cell_measure = jnp.asarray(target)
        self.cell_measure_stiffness = jnp.asarray(measure_stiffness)
        self.target_boundary_measure = jnp.asarray(boundary_target)
        self.boundary_stiffness = jnp.asarray(boundary_modulus)
        self.interface_tension = jnp.asarray(tension)
        self.cell_types = jnp.asarray(types)
        self.adhesion_matrix = jnp.asarray(adhesion)
        self.active_contractility = jnp.asarray(contractility)
        self.cell_traction = jnp.asarray(traction)
        self.vertex_drag = jnp.asarray(drag)
        self.cell_parent_ids = jnp.asarray(parents)
        self.cell_generation = jnp.asarray(generations)
        self.field_names = names
        self.minimum_edge_length = minimum_edge
        self.minimum_cell_measure = minimum_measure
        self.topology_id = topology_id
        self.plan_id = plan_id

    @property
    def vertex_capacity(self) -> int:
        return self.vertex_ids.shape[0]

    @property
    def edge_capacity(self) -> int:
        return self.edge_ids.shape[0]

    @property
    def face_capacity(self) -> int:
        return self.face_ids.shape[0]

    @property
    def cell_capacity(self) -> int:
        return self.cell_ids.shape[0]

    @property
    def field_count(self) -> int:
        return len(self.field_names)

    def prepare(self, reference_positions: ArrayLike, /) -> PreparedVertexTissue:
        """Bind a finite reference geometry to this immutable topology epoch."""

        return PreparedVertexTissue(self, reference_positions)


def polygonal_vertex_tissue_plan(
    vertex_ids: ArrayLike,
    edge_ids: ArrayLike,
    edge_vertex_indices: ArrayLike,
    cell_ids: ArrayLike,
    cell_edge_indices: ArrayLike,
    cell_edge_orientations: ArrayLike,
    interface_cell_indices: ArrayLike,
    target_cell_area: ArrayLike,
    area_stiffness: ArrayLike,
    target_cell_perimeter: ArrayLike,
    perimeter_stiffness: ArrayLike,
    /,
    **kwargs,
) -> VertexTissuePlan:
    """Construct a validated 2D confluent polygonal vertex-tissue plan."""

    return VertexTissuePlan(
        2,
        vertex_ids,
        edge_ids,
        edge_vertex_indices,
        cell_ids,
        interface_cell_indices,
        target_cell_area,
        area_stiffness,
        target_cell_perimeter,
        perimeter_stiffness,
        cell_edge_indices=cell_edge_indices,
        cell_edge_orientations=cell_edge_orientations,
        **kwargs,
    )


def polyhedral_vertex_tissue_plan(
    vertex_ids: ArrayLike,
    edge_ids: ArrayLike,
    edge_vertex_indices: ArrayLike,
    face_ids: ArrayLike,
    face_vertex_indices: ArrayLike,
    cell_ids: ArrayLike,
    cell_face_indices: ArrayLike,
    cell_face_orientations: ArrayLike,
    interface_cell_indices: ArrayLike,
    target_cell_volume: ArrayLike,
    volume_stiffness: ArrayLike,
    target_cell_surface_area: ArrayLike,
    surface_stiffness: ArrayLike,
    /,
    **kwargs,
) -> VertexTissuePlan:
    """Construct a validated 3D confluent polyhedral vertex-tissue plan."""

    return VertexTissuePlan(
        3,
        vertex_ids,
        edge_ids,
        edge_vertex_indices,
        cell_ids,
        interface_cell_indices,
        target_cell_volume,
        volume_stiffness,
        target_cell_surface_area,
        surface_stiffness,
        face_ids=face_ids,
        face_vertex_indices=face_vertex_indices,
        cell_face_indices=cell_face_indices,
        cell_face_orientations=cell_face_orientations,
        **kwargs,
    )


def _cell_vertex_maps(plan: VertexTissuePlan, /) -> tuple[np.ndarray, np.ndarray]:
    incidence = np.zeros((plan.cell_capacity, plan.vertex_capacity), dtype=float)
    if plan.dimension == 2:
        cell_rows = np.asarray(plan.cell_edge_indices)
        edges = np.asarray(plan.edge_vertex_indices)
        for cell in np.flatnonzero(np.asarray(plan.cell_active)):
            for edge in cell_rows[cell][cell_rows[cell] >= 0]:
                incidence[cell, edges[edge]] = 1.0
    else:
        cell_rows = np.asarray(plan.cell_face_indices)
        faces = np.asarray(plan.face_vertex_indices)
        for cell in np.flatnonzero(np.asarray(plan.cell_active)):
            for face in cell_rows[cell][cell_rows[cell] >= 0]:
                vertices = faces[face]
                incidence[cell, vertices[vertices >= 0]] = 1.0
    cell_counts = np.sum(incidence, axis=1, keepdims=True)
    distribution = np.divide(
        incidence,
        cell_counts,
        out=np.zeros_like(incidence),
        where=cell_counts > 0.0,
    )
    vertex_counts = np.sum(incidence, axis=0, keepdims=True).T
    interpolation = np.divide(
        incidence.T,
        vertex_counts,
        out=np.zeros_like(incidence.T),
        where=vertex_counts > 0.0,
    )
    return distribution, interpolation


def _interface_coefficients(plan: VertexTissuePlan, /) -> np.ndarray:
    cells = np.asarray(plan.interface_cell_indices)
    active = np.asarray(plan.edge_active if plan.dimension == 2 else plan.face_active)
    cell_types = np.asarray(plan.cell_types)
    adhesion = np.asarray(plan.adhesion_matrix)
    coefficient = np.asarray(plan.interface_tension).copy()
    for interface in np.flatnonzero(active):
        owners = cells[interface][cells[interface] >= 0]
        if owners.size == 2:
            coefficient[interface] -= adhesion[
                cell_types[owners[0]], cell_types[owners[1]]
            ]
    return coefficient


class VertexTissueState(StrictModule):
    """Fixed-shape vertex coordinates and conserved cell-field contents."""

    positions: Array
    cell_fields: Array
    time: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        positions: ArrayLike,
        cell_fields: ArrayLike,
        time: ArrayLike,
        prepared_id: str,
        /,
    ):
        positions_ = jnp.asarray(positions)
        fields_ = jnp.asarray(cell_fields)
        time_ = jnp.asarray(time)
        if positions_.ndim != 2 or positions_.shape[-1] not in (2, 3):
            raise ValueError("positions must have shape (vertex_capacity, 2|3).")
        if fields_.ndim != 2:
            raise ValueError("cell_fields must have shape (cell_capacity, field_count).")
        if time_.shape != ():
            raise ValueError("time must be scalar.")
        for name, array in (("positions", positions_), ("cell_fields", fields_)):
            if not jnp.issubdtype(array.dtype, jnp.inexact) or jnp.iscomplexobj(array):
                raise TypeError(f"{name} must be a real inexact array.")
        if not jnp.issubdtype(time_.dtype, jnp.inexact) or jnp.iscomplexobj(time_):
            raise TypeError("time must be a real inexact scalar.")
        if (
            not isinstance(prepared_id, str)
            or not prepared_id
            or prepared_id != prepared_id.strip()
        ):
            raise ValueError("prepared_id must be a nonempty canonical identifier.")
        self.positions = positions_
        self.cell_fields = fields_
        self.time = time_
        self.prepared_id = prepared_id


def _vertex_tissue_state_fingerprint(state: VertexTissueState, /) -> str:
    return canonical_fingerprint(
        {
            "kind": "vertex-tissue-state",
            "prepared": state.prepared_id,
            "values": array_tree_fingerprint(
                {
                    "positions": np.asarray(state.positions),
                    "cell_fields": np.asarray(state.cell_fields),
                    "time": np.asarray(state.time),
                }
            ),
        }
    )


class PreparedVertexTissue(StrictModule, NonTrainableState):
    """Prepared incidence maps and interaction routing for one topology epoch."""

    plan: VertexTissuePlan
    reference_positions: Array
    cell_vertex_distribution: Array
    vertex_cell_interpolation: Array
    interface_coefficients: Array
    reference_orientation_valid: Array
    reference_quality_valid: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: VertexTissuePlan, reference_positions: ArrayLike, /):
        if not isinstance(plan, VertexTissuePlan):
            raise TypeError("plan must be a VertexTissuePlan.")
        positions = _real_array("reference_positions", reference_positions, 2)
        expected = (plan.vertex_capacity, plan.dimension)
        if positions.shape != expected:
            raise ValueError(f"reference_positions must have shape {expected}.")
        distribution, interpolation = _cell_vertex_maps(plan)
        coefficients = _interface_coefficients(plan)
        geometry = _vertex_tissue_geometry(plan, jnp.asarray(positions))
        orientation_valid = jnp.all((~plan.cell_active) | (geometry.cell_measure > 0.0))
        quality_valid = (
            geometry.minimum_edge_length > plan.minimum_edge_length
        ) & jnp.all(
            (~plan.cell_active) | (geometry.cell_measure > plan.minimum_cell_measure)
        )
        self.plan = plan
        self.reference_positions = jnp.asarray(positions)
        self.cell_vertex_distribution = jnp.asarray(
            distribution, dtype=self.reference_positions.dtype
        )
        self.vertex_cell_interpolation = jnp.asarray(
            interpolation, dtype=self.reference_positions.dtype
        )
        self.interface_coefficients = jnp.asarray(
            coefficients, dtype=self.reference_positions.dtype
        )
        self.reference_orientation_valid = orientation_valid
        self.reference_quality_valid = quality_valid
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-vertex-tissue",
                "plan": plan.plan_id,
                "reference_geometry": array_tree_fingerprint(positions),
            }
        )

    def initialize_state(
        self,
        cell_fields: ArrayLike | None = None,
        /,
        *,
        time: ArrayLike = 0.0,
    ) -> VertexTissueState:
        fields = (
            jnp.zeros(
                (self.plan.cell_capacity, self.plan.field_count),
                dtype=self.reference_positions.dtype,
            )
            if cell_fields is None
            else jnp.asarray(cell_fields, dtype=self.reference_positions.dtype)
        )
        expected = (self.plan.cell_capacity, self.plan.field_count)
        if fields.shape != expected:
            raise ValueError(f"cell_fields must have shape {expected}.")
        if np.any(np.asarray(fields)[~np.asarray(self.plan.cell_active)] != 0.0):
            raise ValueError("Inactive cell field slots must be zero.")
        return VertexTissueState(self.reference_positions, fields, time, self.prepared_id)

    def evaluate(self, state: VertexTissueState, /) -> VertexTissueEvaluation:
        return evaluate_vertex_tissue(self, state)

    def potential_energy(self, positions: ArrayLike, /) -> Array:
        return vertex_tissue_potential_energy(self, positions)

    def interpolate_cell_fields(self, state: VertexTissueState, /) -> Array:
        """Interpolate intensive cell fields to vertices by incidence averaging."""

        _validate_state(self, state)
        return oe.contract("vc,cf->vf", self.vertex_cell_interpolation, state.cell_fields)

    def spread_vertex_field_sources(self, vertex_sources: ArrayLike, /) -> Array:
        """Apply the exact transpose of cell-to-vertex field interpolation."""

        sources = jnp.asarray(vertex_sources)
        expected = (self.plan.vertex_capacity, self.plan.field_count)
        if sources.shape != expected:
            raise ValueError(f"vertex_sources must have shape {expected}.")
        return oe.contract("vc,vf->cf", self.vertex_cell_interpolation, sources)


class _VertexTissueGeometry(StrictModule):
    cell_measure: Array
    boundary_measure: Array
    interface_measure: Array
    edge_lengths: Array
    minimum_edge_length: Array


def _vertex_tissue_geometry(
    plan: VertexTissuePlan, positions: Array, /
) -> _VertexTissueGeometry:
    edge_indices = jnp.clip(plan.edge_vertex_indices, 0, plan.vertex_capacity - 1)
    edge_vectors = positions[edge_indices[:, 1]] - positions[edge_indices[:, 0]]
    edge_squared = jnp.sum(edge_vectors * edge_vectors, axis=-1)
    safe_edge_squared = jnp.maximum(edge_squared, jnp.finfo(positions.dtype).tiny)
    edge_lengths = jnp.where(plan.edge_active, jnp.sqrt(safe_edge_squared), 0.0)
    minimum_edge = jnp.min(jnp.where(plan.edge_active, edge_lengths, jnp.inf))
    if plan.dimension == 2:
        safe_edges = jnp.clip(plan.cell_edge_indices, 0, plan.edge_capacity - 1)
        valid = plan.cell_edge_indices >= 0
        endpoints = plan.edge_vertex_indices[safe_edges]
        first = positions[endpoints[..., 0]]
        second = positions[endpoints[..., 1]]
        cross = first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]
        signed = plan.cell_edge_orientations * cross
        cell_measure = 0.5 * jnp.sum(jnp.where(valid, signed, 0.0), axis=1)
        boundary = jnp.sum(jnp.where(valid, edge_lengths[safe_edges], 0.0), axis=1)
        interface = edge_lengths
    else:
        safe_vertices = jnp.clip(plan.face_vertex_indices, 0, plan.vertex_capacity - 1)
        face_points = positions[safe_vertices]
        current_valid = plan.face_vertex_indices >= 0
        shifted_indices = jnp.roll(plan.face_vertex_indices, -1, axis=1)
        next_indices = jnp.where(
            current_valid & (shifted_indices < 0),
            plan.face_vertex_indices[:, :1],
            shifted_indices,
        )
        safe_next = jnp.clip(next_indices, 0, plan.vertex_capacity - 1)
        area_vector = 0.5 * jnp.sum(
            jnp.where(
                current_valid[..., None],
                jnp.cross(face_points, positions[safe_next]),
                0.0,
            ),
            axis=1,
        )
        face_area_squared = jnp.sum(area_vector * area_vector, axis=-1)
        safe_face_area_squared = jnp.maximum(
            face_area_squared, jnp.finfo(positions.dtype).tiny
        )
        face_area = jnp.where(plan.face_active, jnp.sqrt(safe_face_area_squared), 0.0)
        triangle_valid = (
            (plan.face_vertex_indices[:, :1] >= 0)
            & (plan.face_vertex_indices[:, 1:-1] >= 0)
            & (plan.face_vertex_indices[:, 2:] >= 0)
        )
        first = face_points[:, :1, :]
        second = face_points[:, 1:-1, :]
        third = face_points[:, 2:, :]
        cross = jnp.cross(second - first, third - first)
        flux = jnp.sum(
            jnp.where(
                triangle_valid,
                oe.contract("fti,fti->ft", first + jnp.zeros_like(cross), cross) / 6.0,
                0.0,
            ),
            axis=1,
        )
        safe_faces = jnp.clip(plan.cell_face_indices, 0, plan.face_capacity - 1)
        valid = plan.cell_face_indices >= 0
        cell_measure = jnp.sum(
            jnp.where(
                valid,
                plan.cell_face_orientations * flux[safe_faces],
                0.0,
            ),
            axis=1,
        )
        boundary = jnp.sum(jnp.where(valid, face_area[safe_faces], 0.0), axis=1)
        interface = face_area
    cell_measure = jnp.where(plan.cell_active, cell_measure, 0.0)
    boundary = jnp.where(plan.cell_active, boundary, 0.0)
    return _VertexTissueGeometry(
        cell_measure, boundary, interface, edge_lengths, minimum_edge
    )


def vertex_tissue_potential_energy(
    prepared: PreparedVertexTissue, positions: ArrayLike, /
) -> Array:
    """Return the objective conservative scalar energy for one tissue geometry."""

    if not isinstance(prepared, PreparedVertexTissue):
        raise TypeError("prepared must be a PreparedVertexTissue.")
    positions_ = jnp.asarray(positions)
    expected = (prepared.plan.vertex_capacity, prepared.plan.dimension)
    if positions_.shape != expected:
        raise ValueError(f"positions must have shape {expected}.")
    plan = prepared.plan
    geometry = _vertex_tissue_geometry(plan, positions_)
    cell_mask = plan.cell_active
    measure_energy = 0.5 * jnp.sum(
        jnp.where(
            cell_mask,
            plan.cell_measure_stiffness
            * (geometry.cell_measure - plan.target_cell_measure) ** 2,
            0.0,
        )
    )
    boundary_energy = 0.5 * jnp.sum(
        jnp.where(
            cell_mask,
            plan.boundary_stiffness
            * (geometry.boundary_measure - plan.target_boundary_measure) ** 2,
            0.0,
        )
    )
    contractile_energy = 0.5 * jnp.sum(
        jnp.where(
            cell_mask,
            plan.active_contractility * geometry.boundary_measure**2,
            0.0,
        )
    )
    interface_active = plan.edge_active if plan.dimension == 2 else plan.face_active
    interface_energy = jnp.sum(
        jnp.where(
            interface_active,
            prepared.interface_coefficients * geometry.interface_measure,
            0.0,
        )
    )
    return measure_energy + boundary_energy + contractile_energy + interface_energy


class VertexTissueEvaluation(StrictModule):
    """Energy, loads, cell observables, and finite-domain evidence."""

    cell_measure: Array
    boundary_measure: Array
    interface_measure: Array
    adhesion_routed_tension: Array
    cell_field_density: Array
    measure_energy: Array
    boundary_energy: Array
    contractile_energy: Array
    interface_energy: Array
    potential_energy: Array
    conservative_forces: Array
    active_forces: Array
    total_forces: Array
    net_conservative_force_residual: Array
    minimum_edge_length: Array
    minimum_cell_measure: Array
    inactive_fields_valid: Array
    finite: Array
    manifold: Array
    orientation_valid: Array
    quality_valid: Array
    valid: Array
    status: Array
    prepared_id: str = eqx.field(static=True)


def _validate_state(prepared: PreparedVertexTissue, state: VertexTissueState, /) -> None:
    if not isinstance(state, VertexTissueState):
        raise TypeError("state must be a VertexTissueState.")
    plan = prepared.plan
    if state.prepared_id != prepared.prepared_id:
        raise ValueError("State belongs to a different prepared topology epoch.")
    if state.positions.shape != (plan.vertex_capacity, plan.dimension):
        raise ValueError("State position shape does not match the prepared tissue.")
    if state.cell_fields.shape != (plan.cell_capacity, plan.field_count):
        raise ValueError("State field shape does not match the prepared tissue.")


def evaluate_vertex_tissue(
    prepared: PreparedVertexTissue, state: VertexTissueState, /
) -> VertexTissueEvaluation:
    """Evaluate conservative and active vertex-tissue mechanics."""

    if not isinstance(prepared, PreparedVertexTissue):
        raise TypeError("prepared must be a PreparedVertexTissue.")
    _validate_state(prepared, state)
    plan = prepared.plan
    geometry = _vertex_tissue_geometry(plan, state.positions)
    potential, gradient = jax.value_and_grad(vertex_tissue_potential_energy, argnums=1)(
        prepared, state.positions
    )
    raw_conservative = -gradient
    active_forces = oe.contract(
        "cv,cd->vd", prepared.cell_vertex_distribution, plan.cell_traction
    )
    cell_mask = plan.cell_active
    measure_terms = jnp.where(
        cell_mask,
        0.5
        * plan.cell_measure_stiffness
        * (geometry.cell_measure - plan.target_cell_measure) ** 2,
        0.0,
    )
    boundary_terms = jnp.where(
        cell_mask,
        0.5
        * plan.boundary_stiffness
        * (geometry.boundary_measure - plan.target_boundary_measure) ** 2,
        0.0,
    )
    contractile_terms = jnp.where(
        cell_mask,
        0.5 * plan.active_contractility * geometry.boundary_measure**2,
        0.0,
    )
    interface_active = plan.edge_active if plan.dimension == 2 else plan.face_active
    interface_terms = jnp.where(
        interface_active,
        prepared.interface_coefficients * geometry.interface_measure,
        0.0,
    )
    safe_measure = jnp.maximum(jnp.abs(geometry.cell_measure), plan.minimum_cell_measure)
    density = jnp.where(
        cell_mask[:, None], state.cell_fields / safe_measure[:, None], 0.0
    )
    minimum_cell_measure = jnp.min(jnp.where(cell_mask, geometry.cell_measure, jnp.inf))
    orientation_valid = jnp.all((~cell_mask) | (geometry.cell_measure > 0.0))
    quality_valid = (geometry.minimum_edge_length > plan.minimum_edge_length) & (
        minimum_cell_measure > plan.minimum_cell_measure
    )
    inactive_fields_valid = jnp.all(
        jnp.where(cell_mask[:, None], True, state.cell_fields == 0.0)
    )
    finite = (
        jnp.all(jnp.isfinite(state.positions))
        & jnp.all(jnp.isfinite(state.cell_fields))
        & jnp.isfinite(state.time)
        & jnp.all(jnp.isfinite(geometry.cell_measure))
        & jnp.all(jnp.isfinite(geometry.boundary_measure))
        & jnp.all(jnp.isfinite(geometry.interface_measure))
        & jnp.isfinite(potential)
        & jnp.all(jnp.isfinite(raw_conservative))
        & jnp.all(jnp.isfinite(active_forces))
        & jnp.all(jnp.isfinite(density))
    )
    manifold = jnp.asarray(True)
    valid = finite & manifold & orientation_valid & quality_valid & inactive_fields_valid
    conservative = jnp.where(valid, raw_conservative, 0.0)
    active = jnp.where(valid, active_forces, 0.0)
    total = conservative + active
    net_residual = jnp.sqrt(jnp.sum(jnp.sum(conservative, axis=0) ** 2))
    status = jnp.asarray(int(VertexTissueStatus.SUCCESS), dtype=jnp.int32)
    status = jnp.where(
        ~quality_valid,
        int(VertexTissueStatus.QUALITY_FAILURE),
        status,
    )
    status = jnp.where(
        ~orientation_valid,
        int(VertexTissueStatus.ORIENTATION_FAILURE),
        status,
    )
    status = jnp.where(~manifold, int(VertexTissueStatus.NONMANIFOLD), status)
    status = jnp.where(
        ~inactive_fields_valid,
        int(VertexTissueStatus.CONSERVATION_FAILURE),
        status,
    )
    status = jnp.where(~finite, int(VertexTissueStatus.NONFINITE), status)
    return VertexTissueEvaluation(
        geometry.cell_measure,
        geometry.boundary_measure,
        geometry.interface_measure,
        prepared.interface_coefficients,
        density,
        jnp.sum(measure_terms),
        jnp.sum(boundary_terms),
        jnp.sum(contractile_terms),
        jnp.sum(interface_terms),
        potential,
        conservative,
        active,
        total,
        net_residual,
        geometry.minimum_edge_length,
        minimum_cell_measure,
        inactive_fields_valid,
        finite,
        manifold,
        orientation_valid,
        quality_valid,
        valid,
        status,
        prepared.prepared_id,
    )


class VertexTissueParticleCoupling(StrictModule):
    """Cell-field gather and conservative particle-force scatter evidence."""

    particle_fields: Array
    vertex_forces: Array
    force_conservation_residual: Array
    route_valid: Array
    finite: Array
    valid: Array
    status: Array


def couple_vertex_tissue_particles(
    prepared: PreparedVertexTissue,
    state: VertexTissueState,
    particle_cell_indices: ArrayLike,
    particle_forces: ArrayLike,
    /,
) -> VertexTissueParticleCoupling:
    """Gather cell fields to particles and scatter resultants to cell vertices.

    ``particle_cell_indices == -1`` denotes a masked particle. Each active
    particle force is distributed uniformly over the distinct vertices incident
    to its owning cell, preserving total force.
    """

    if not isinstance(prepared, PreparedVertexTissue):
        raise TypeError("prepared must be a PreparedVertexTissue.")
    _validate_state(prepared, state)
    cells = jnp.asarray(particle_cell_indices)
    forces = jnp.asarray(particle_forces)
    if cells.ndim != 1 or not jnp.issubdtype(cells.dtype, jnp.integer):
        raise TypeError("particle_cell_indices must be an integer vector.")
    expected = (cells.shape[0], prepared.plan.dimension)
    if forces.shape != expected:
        raise ValueError(f"particle_forces must have shape {expected}.")
    safe = jnp.clip(cells, 0, prepared.plan.cell_capacity - 1)
    active = cells >= 0
    route_valid = jnp.all(
        (cells == -1)
        | (
            (cells >= 0)
            & (cells < prepared.plan.cell_capacity)
            & prepared.plan.cell_active[safe]
        )
    )
    gathered = jnp.where(active[:, None], state.cell_fields[safe], 0.0)
    routed_force = jnp.where(active[:, None], forces, 0.0)
    cell_force = (
        jnp.zeros(
            (prepared.plan.cell_capacity, prepared.plan.dimension), dtype=forces.dtype
        )
        .at[safe]
        .add(routed_force)
    )
    vertex_force = oe.contract("cv,cd->vd", prepared.cell_vertex_distribution, cell_force)
    residual = jnp.sqrt(
        jnp.sum((jnp.sum(vertex_force, axis=0) - jnp.sum(routed_force, axis=0)) ** 2)
    )
    finite = (
        jnp.all(jnp.isfinite(forces))
        & jnp.all(jnp.isfinite(gathered))
        & jnp.all(jnp.isfinite(vertex_force))
        & jnp.isfinite(residual)
    )
    valid = route_valid & finite
    status = jnp.where(
        ~finite,
        int(VertexTissueStatus.NONFINITE),
        jnp.where(
            ~route_valid,
            int(VertexTissueStatus.INVALID_EVENT),
            int(VertexTissueStatus.SUCCESS),
        ),
    ).astype(jnp.int32)
    return VertexTissueParticleCoupling(
        jnp.where(valid, gathered, 0.0),
        jnp.where(valid, vertex_force, 0.0),
        residual,
        route_valid,
        finite,
        valid,
        status,
    )


class VertexTissueDynamicsPlan(StrictModule, NonTrainableState):
    """Bounded explicit overdamped evolution policy for a prepared tissue epoch."""

    step_size: float = eqx.field(static=True)
    maximum_displacement: float = eqx.field(static=True)
    energy_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        step_size: float,
        /,
        *,
        maximum_displacement: float = 1.0,
        energy_tolerance: float = 1.0e-10,
    ):
        step = float(step_size)
        maximum = float(maximum_displacement)
        tolerance = float(energy_tolerance)
        if (
            not np.isfinite(step)
            or step <= 0.0
            or not np.isfinite(maximum)
            or maximum <= 0.0
            or not np.isfinite(tolerance)
            or tolerance < 0.0
        ):
            raise ValueError(
                "Dynamics controls must be finite and in their positive domains."
            )
        self.step_size = step
        self.maximum_displacement = maximum
        self.energy_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "overdamped-vertex-tissue-dynamics",
                "step_size": step,
                "maximum_displacement": maximum,
                "energy_tolerance": tolerance,
            }
        )

    def prepare(self, tissue: PreparedVertexTissue, /) -> PreparedVertexTissueDynamics:
        return PreparedVertexTissueDynamics(self, tissue)


class PreparedVertexTissueDynamics(StrictModule, NonTrainableState):
    """JIT-compatible overdamped step bound to one prepared topology epoch."""

    plan: VertexTissueDynamicsPlan
    tissue: PreparedVertexTissue
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: VertexTissueDynamicsPlan, tissue: PreparedVertexTissue, /):
        if not isinstance(plan, VertexTissueDynamicsPlan):
            raise TypeError("plan must be a VertexTissueDynamicsPlan.")
        if not isinstance(tissue, PreparedVertexTissue):
            raise TypeError("tissue must be a PreparedVertexTissue.")
        self.plan = plan
        self.tissue = tissue
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-overdamped-vertex-tissue-dynamics",
                "plan": plan.plan_id,
                "tissue": tissue.prepared_id,
            }
        )

    def step(
        self,
        state: VertexTissueState,
        /,
        *,
        external_vertex_forces: ArrayLike | None = None,
        cell_field_rates: ArrayLike | None = None,
    ) -> VertexTissueStepResult:
        return step_vertex_tissue_overdamped(
            self,
            state,
            external_vertex_forces=external_vertex_forces,
            cell_field_rates=cell_field_rates,
        )


class VertexTissueStepResult(StrictModule):
    """Candidate/accepted overdamped state and energetic transition evidence."""

    state: VertexTissueState
    candidate_state: VertexTissueState
    before: VertexTissueEvaluation
    after: VertexTissueEvaluation
    velocity: Array
    dissipation_rate: Array
    active_power: Array
    energy_change: Array
    energy_descent: Array
    displacement_valid: Array
    field_rate_valid: Array
    finite: Array
    accepted: Array
    status: Array
    dynamics_id: str = eqx.field(static=True)


def step_vertex_tissue_overdamped(
    prepared: PreparedVertexTissueDynamics,
    state: VertexTissueState,
    /,
    *,
    external_vertex_forces: ArrayLike | None = None,
    cell_field_rates: ArrayLike | None = None,
) -> VertexTissueStepResult:
    """Advance one pure fixed-shape overdamped candidate and atomically accept it."""

    if not isinstance(prepared, PreparedVertexTissueDynamics):
        raise TypeError("prepared must be a PreparedVertexTissueDynamics.")
    tissue = prepared.tissue
    _validate_state(tissue, state)
    plan = tissue.plan
    external = (
        jnp.zeros_like(state.positions)
        if external_vertex_forces is None
        else jnp.asarray(external_vertex_forces, dtype=state.positions.dtype)
    )
    rates = (
        jnp.zeros_like(state.cell_fields)
        if cell_field_rates is None
        else jnp.asarray(cell_field_rates, dtype=state.cell_fields.dtype)
    )
    if external.shape != state.positions.shape:
        raise ValueError("external_vertex_forces must match state positions.")
    if rates.shape != state.cell_fields.shape:
        raise ValueError("cell_field_rates must match state cell_fields.")
    field_rate_valid = jnp.all(jnp.where(plan.cell_active[:, None], True, rates == 0.0))
    before = evaluate_vertex_tissue(tissue, state)
    applied = before.total_forces + external
    velocity = applied / plan.vertex_drag[:, None]
    velocity = jnp.where(plan.vertex_active[:, None], velocity, 0.0)
    displacement = prepared.plan.step_size * velocity
    candidate = VertexTissueState(
        state.positions + displacement,
        state.cell_fields + prepared.plan.step_size * rates,
        state.time + prepared.plan.step_size,
        tissue.prepared_id,
    )
    after = evaluate_vertex_tissue(tissue, candidate)
    maximum_displacement = jnp.max(
        jnp.sqrt(jnp.sum(displacement * displacement, axis=-1))
    )
    displacement_valid = maximum_displacement <= prepared.plan.maximum_displacement
    dissipation = jnp.sum(plan.vertex_drag[:, None] * velocity * velocity)
    active_power = jnp.sum((before.active_forces + external) * velocity)
    energy_change = after.potential_energy - before.potential_energy
    passive = jnp.all(before.active_forces == 0.0) & jnp.all(external == 0.0)
    energy_descent = (~passive) | (energy_change <= prepared.plan.energy_tolerance)
    finite = (
        before.finite
        & after.finite
        & jnp.all(jnp.isfinite(external))
        & jnp.all(jnp.isfinite(rates))
        & jnp.all(jnp.isfinite(velocity))
        & jnp.isfinite(dissipation)
        & jnp.isfinite(active_power)
        & jnp.isfinite(energy_change)
    )
    accepted = (
        before.valid
        & after.valid
        & finite
        & displacement_valid
        & energy_descent
        & field_rate_valid
    )
    accepted_state = VertexTissueState(
        jnp.where(accepted, candidate.positions, state.positions),
        jnp.where(accepted, candidate.cell_fields, state.cell_fields),
        jnp.where(accepted, candidate.time, state.time),
        tissue.prepared_id,
    )
    status = jnp.asarray(int(VertexTissueStatus.SUCCESS), dtype=jnp.int32)
    status = jnp.where(
        ~energy_descent | ~after.quality_valid,
        int(VertexTissueStatus.QUALITY_FAILURE),
        status,
    )
    status = jnp.where(
        ~displacement_valid,
        int(VertexTissueStatus.CAPACITY_EXCEEDED),
        status,
    )
    status = jnp.where(
        ~field_rate_valid,
        int(VertexTissueStatus.CONSERVATION_FAILURE),
        status,
    )
    status = jnp.where(~finite, int(VertexTissueStatus.NONFINITE), status)
    return VertexTissueStepResult(
        accepted_state,
        candidate,
        before,
        after,
        velocity,
        dissipation,
        active_power,
        energy_change,
        energy_descent,
        displacement_valid,
        field_rate_valid,
        finite,
        accepted,
        status,
        prepared.prepared_id,
    )


class VertexTissueTopologyEvent(StrictModule, NonTrainableState):
    """Host-side request carrying a complete replacement epoch and field map."""

    kind: VertexTissueEventKind = eqx.field(static=True)
    source_prepared_id: str = eqx.field(static=True)
    target_plan: VertexTissuePlan
    target_positions: Array
    cell_transfer: Array
    conservation_tolerance: float = eqx.field(static=True)
    event_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: VertexTissueEventKind | int,
        source_prepared_id: str,
        target_plan: VertexTissuePlan,
        target_positions: ArrayLike,
        cell_transfer: ArrayLike,
        /,
        *,
        conservation_tolerance: float = 1.0e-8,
    ):
        kind_ = _enum_kind(kind)
        if (
            not isinstance(source_prepared_id, str)
            or not source_prepared_id
            or source_prepared_id != source_prepared_id.strip()
        ):
            raise ValueError(
                "source_prepared_id must be a nonempty canonical identifier."
            )
        if not isinstance(target_plan, VertexTissuePlan):
            raise TypeError("target_plan must be a VertexTissuePlan.")
        positions = _real_array("target_positions", target_positions, 2)
        if positions.shape != (target_plan.vertex_capacity, target_plan.dimension):
            raise ValueError("target_positions do not match target_plan capacity.")
        transfer = _real_array("cell_transfer", cell_transfer, 2)
        if transfer.shape[0] != target_plan.cell_capacity:
            raise ValueError("cell_transfer rows must match target cell capacity.")
        tolerance = float(conservation_tolerance)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("conservation_tolerance must be finite and nonnegative.")
        self.kind = kind_
        self.source_prepared_id = source_prepared_id
        self.target_plan = target_plan
        self.target_positions = jnp.asarray(positions)
        self.cell_transfer = jnp.asarray(transfer)
        self.conservation_tolerance = tolerance
        self.event_id = canonical_fingerprint(
            {
                "kind": "vertex-tissue-topology-event",
                "event_kind": int(kind_),
                "source": source_prepared_id,
                "target": target_plan.plan_id,
                "positions": array_tree_fingerprint(positions),
                "transfer": array_tree_fingerprint(transfer),
                "conservation_tolerance": tolerance,
            }
        )


class VertexTissueTopologyCandidate(StrictModule, NonTrainableState):
    """Uncommitted replacement prepared epoch and conservatively mapped state."""

    event: VertexTissueTopologyEvent
    prepared: PreparedVertexTissue
    state: VertexTissueState
    source_state_id: str = eqx.field(static=True)
    candidate_id: str = eqx.field(static=True)

    def __init__(
        self,
        event: VertexTissueTopologyEvent,
        prepared: PreparedVertexTissue,
        state: VertexTissueState,
        source_state_id: str,
        /,
    ):
        if not isinstance(event, VertexTissueTopologyEvent):
            raise TypeError("event must be a VertexTissueTopologyEvent.")
        if not isinstance(prepared, PreparedVertexTissue):
            raise TypeError("prepared must be a PreparedVertexTissue.")
        if not isinstance(state, VertexTissueState):
            raise TypeError("state must be a VertexTissueState.")
        if prepared.plan.plan_id != event.target_plan.plan_id:
            raise ValueError("Candidate prepared plan does not match its event.")
        _validate_state(prepared, state)
        if (
            not isinstance(source_state_id, str)
            or not source_state_id
            or source_state_id != source_state_id.strip()
        ):
            raise ValueError("source_state_id must be a nonempty canonical identifier.")
        self.event = event
        self.prepared = prepared
        self.state = state
        self.source_state_id = source_state_id
        self.candidate_id = canonical_fingerprint(
            {
                "kind": "vertex-tissue-topology-candidate",
                "event": event.event_id,
                "prepared": prepared.prepared_id,
                "source_state": source_state_id,
                "state": _vertex_tissue_state_fingerprint(state),
            }
        )


def propose_vertex_tissue_topology(
    source: PreparedVertexTissue,
    state: VertexTissueState,
    event: VertexTissueTopologyEvent,
    /,
) -> VertexTissueTopologyCandidate:
    """Materialize, but never commit, one fixed-capacity topology replacement."""

    if not isinstance(source, PreparedVertexTissue):
        raise TypeError("source must be a PreparedVertexTissue.")
    _validate_state(source, state)
    if not isinstance(event, VertexTissueTopologyEvent):
        raise TypeError("event must be a VertexTissueTopologyEvent.")
    expected = (event.target_plan.cell_capacity, source.plan.cell_capacity)
    if event.cell_transfer.shape != expected:
        raise ValueError(f"cell_transfer must have shape {expected}.")
    if event.target_plan.field_names != source.plan.field_names:
        raise ValueError("Target and source field identities must agree.")
    prepared = event.target_plan.prepare(event.target_positions)
    fields = oe.contract("ts,sf->tf", event.cell_transfer, state.cell_fields)
    target_state = VertexTissueState(
        event.target_positions, fields, state.time, prepared.prepared_id
    )
    return VertexTissueTopologyCandidate(
        event, prepared, target_state, _vertex_tissue_state_fingerprint(state)
    )


def _active_id_set(values: Array, /) -> set[int]:
    host = np.asarray(values)
    return set(int(value) for value in host[host >= 0])


def _cell_slot_by_id(plan: VertexTissuePlan, stable_id: int, /) -> int:
    identifiers = np.asarray(plan.cell_ids)
    matches = np.flatnonzero(identifiers == stable_id)
    return -1 if matches.size == 0 else int(matches[0])


def _unordered_rows(values: Array, /) -> tuple[tuple[int, ...], ...]:
    host = np.asarray(values)
    return tuple(tuple(sorted(int(value) for value in row if value >= 0)) for row in host)


def _edge_incidence_signature(plan: VertexTissuePlan, /) -> tuple[tuple[int, ...], ...]:
    return _unordered_rows(plan.edge_vertex_indices)


def _face_incidence_signature(plan: VertexTissuePlan, /) -> tuple[tuple[int, ...], ...]:
    host = np.asarray(plan.face_vertex_indices)
    return tuple(
        () if not np.any(row >= 0) else _canonical_cycle(row[row >= 0]) for row in host
    )


def _t1_exchange_valid(source: VertexTissuePlan, target: VertexTissuePlan, /) -> bool:
    if source.dimension != 2 or target.dimension != 2:
        return False
    stable_slots = (
        np.array_equal(np.asarray(source.vertex_ids), np.asarray(target.vertex_ids))
        and np.array_equal(np.asarray(source.edge_ids), np.asarray(target.edge_ids))
        and np.array_equal(np.asarray(source.cell_ids), np.asarray(target.cell_ids))
    )
    if not stable_slots:
        return False
    source_owners = _unordered_rows(source.interface_cell_indices)
    target_owners = _unordered_rows(target.interface_cell_indices)
    changed = tuple(
        index
        for index, (before, after) in enumerate(zip(source_owners, target_owners))
        if before != after
    )
    if len(changed) != 1:
        return False
    edge = changed[0]
    before = source_owners[edge]
    after = target_owners[edge]
    four_cell_exchange = (
        len(before) == 2
        and len(after) == 2
        and set(before).isdisjoint(after)
        and len(set(before) | set(after)) == 4
    )
    cell_incidence_changed = _unordered_rows(source.cell_edge_indices) != _unordered_rows(
        target.cell_edge_indices
    )
    return bool(
        four_cell_exchange
        and cell_incidence_changed
        and int(np.asarray(source.edge_ids)[edge]) >= 0
        and int(np.asarray(source.edge_ids)[edge])
        == int(np.asarray(target.edge_ids)[edge])
    )


def _kind_and_lineage_valid(
    source: VertexTissuePlan,
    target: VertexTissuePlan,
    kind: VertexTissueEventKind,
    /,
) -> tuple[bool, bool]:
    if (
        source.dimension != target.dimension
        or source.vertex_capacity != target.vertex_capacity
        or source.edge_capacity != target.edge_capacity
        or source.face_capacity != target.face_capacity
        or source.cell_capacity != target.cell_capacity
    ):
        return False, False
    source_ids = _active_id_set(source.cell_ids)
    target_ids = _active_id_set(target.cell_ids)
    added = target_ids - source_ids
    removed = source_ids - target_ids
    topology_changed = source.topology_id != target.topology_id
    edge_delta = _edge_incidence_signature(source) != _edge_incidence_signature(target)
    interface_delta = _unordered_rows(source.interface_cell_indices) != _unordered_rows(
        target.interface_cell_indices
    )
    if source.dimension == 2:
        cell_incidence_delta = _unordered_rows(
            source.cell_edge_indices
        ) != _unordered_rows(target.cell_edge_indices)
        t1_exchange = _t1_exchange_valid(source, target)
    else:
        cell_incidence_delta = False
        t1_exchange = False

    if kind is VertexTissueEventKind.T1:
        kind_valid = not added and not removed and topology_changed and t1_exchange
    elif kind is VertexTissueEventKind.T2:
        triangular = False
        if source.dimension == 2 and len(removed) == 1:
            slot = _cell_slot_by_id(source, next(iter(removed)))
            triangular = int(np.sum(np.asarray(source.cell_edge_indices)[slot] >= 0)) == 3
        kind_valid = (
            source.dimension == 2
            and not added
            and len(removed) == 1
            and triangular
            and topology_changed
        )
    elif kind is VertexTissueEventKind.T3:
        stable_slots = (
            np.array_equal(np.asarray(source.vertex_ids), np.asarray(target.vertex_ids))
            and np.array_equal(np.asarray(source.edge_ids), np.asarray(target.edge_ids))
            and np.array_equal(np.asarray(source.cell_ids), np.asarray(target.cell_ids))
        )
        actual_vertex_edge_change = edge_delta or cell_incidence_delta or interface_delta
        kind_valid = (
            source.dimension == 2
            and not added
            and not removed
            and topology_changed
            and stable_slots
            and actual_vertex_edge_change
            and not t1_exchange
        )
    elif kind is VertexTissueEventKind.DIVISION:
        kind_valid = len(added) == 1 and not removed and topology_changed
    elif kind in (
        VertexTissueEventKind.EXTRUSION,
        VertexTissueEventKind.APOPTOSIS,
    ):
        kind_valid = not added and len(removed) == 1 and topology_changed
    else:
        stable_cells = np.array_equal(
            np.asarray(source.cell_ids), np.asarray(target.cell_ids)
        )
        face_delta = (
            _face_incidence_signature(source) != _face_incidence_signature(target)
            or _unordered_rows(source.cell_face_indices)
            != _unordered_rows(target.cell_face_indices)
            or interface_delta
        )
        common = (
            source.dimension == 3
            and not added
            and not removed
            and topology_changed
            and stable_cells
        )
        if kind is VertexTissueEventKind.FACE_TRANSITION:
            kind_valid = common and face_delta and not edge_delta
        else:
            kind_valid = common and edge_delta and face_delta

    lineage_valid = True
    for stable_id in source_ids & target_ids:
        source_slot = _cell_slot_by_id(source, stable_id)
        target_slot = _cell_slot_by_id(target, stable_id)
        lineage_valid = lineage_valid and (
            int(np.asarray(source.cell_parent_ids)[source_slot])
            == int(np.asarray(target.cell_parent_ids)[target_slot])
            and int(np.asarray(source.cell_generation)[source_slot])
            == int(np.asarray(target.cell_generation)[target_slot])
        )
    if added:
        if kind is not VertexTissueEventKind.DIVISION:
            lineage_valid = False
        for stable_id in added:
            target_slot = _cell_slot_by_id(target, stable_id)
            parent_id = int(np.asarray(target.cell_parent_ids)[target_slot])
            parent_slot = _cell_slot_by_id(source, parent_id)
            lineage_valid = lineage_valid and parent_slot >= 0
            if parent_slot >= 0:
                lineage_valid = lineage_valid and (
                    int(np.asarray(target.cell_generation)[target_slot])
                    == int(np.asarray(source.cell_generation)[parent_slot]) + 1
                )
    return bool(kind_valid), bool(lineage_valid)


def _lineage_transfer_valid(
    source: VertexTissuePlan,
    target: VertexTissuePlan,
    kind: VertexTissueEventKind,
    transfer: Array,
    tolerance: float,
    /,
) -> bool:
    if source.cell_capacity != target.cell_capacity:
        return False
    source_ids = _active_id_set(source.cell_ids)
    target_ids = _active_id_set(target.cell_ids)
    added = target_ids - source_ids
    removed = source_ids - target_ids
    matrix = np.asarray(transfer)
    allowed = np.zeros_like(matrix, dtype=bool)
    for stable_id in source_ids & target_ids:
        source_slot = _cell_slot_by_id(source, stable_id)
        target_slot = _cell_slot_by_id(target, stable_id)
        allowed[target_slot, source_slot] = True
        if stable_id not in {
            int(np.asarray(target.cell_parent_ids)[_cell_slot_by_id(target, added_id)])
            for added_id in added
        }:
            if abs(float(matrix[target_slot, source_slot]) - 1.0) > tolerance:
                return False
    if kind is VertexTissueEventKind.DIVISION:
        if len(added) != 1:
            return False
        daughter_slot = _cell_slot_by_id(target, next(iter(added)))
        parent_id = int(np.asarray(target.cell_parent_ids)[daughter_slot])
        parent_slot = _cell_slot_by_id(source, parent_id)
        if parent_slot < 0 or float(matrix[daughter_slot, parent_slot]) <= tolerance:
            return False
        allowed[daughter_slot, parent_slot] = True
    elif removed:
        removed_slots = tuple(
            _cell_slot_by_id(source, stable_id) for stable_id in removed
        )
        for target_slot in np.flatnonzero(np.asarray(target.cell_active)):
            for source_slot in removed_slots:
                allowed[target_slot, source_slot] = True
    return bool(np.all(np.abs(matrix[~allowed]) <= tolerance))


class VertexTissueTopologyEvaluation(StrictModule):
    """Complete fail-closed evidence for one uncommitted topology candidate."""

    target: VertexTissueEvaluation
    field_conservation_defect: Array
    fresh: Array
    source_state_valid: Array
    mapping_valid: Array
    capacity_valid: Array
    kind_valid: Array
    manifold: Array
    orientation_valid: Array
    quality_valid: Array
    transfer_valid: Array
    conservation_valid: Array
    lineage_transfer_valid: Array
    lineage_valid: Array
    finite: Array
    passed: Array
    status: Array
    candidate_id: str = eqx.field(static=True)


def evaluate_vertex_tissue_topology(
    source: PreparedVertexTissue,
    source_state: VertexTissueState,
    candidate: VertexTissueTopologyCandidate,
    /,
) -> VertexTissueTopologyEvaluation:
    """Certify identity, capacity, geometry, transfer, and lineage before commit."""

    if not isinstance(source, PreparedVertexTissue):
        raise TypeError("source must be a PreparedVertexTissue.")
    _validate_state(source, source_state)
    if not isinstance(candidate, VertexTissueTopologyCandidate):
        raise TypeError("candidate must be a VertexTissueTopologyCandidate.")
    source_evaluation = evaluate_vertex_tissue(source, source_state)
    event = candidate.event
    target_plan = candidate.prepared.plan
    target = evaluate_vertex_tissue(candidate.prepared, candidate.state)
    fresh = jnp.asarray(event.source_prepared_id == source.prepared_id)
    source_state_valid = (
        jnp.asarray(
            candidate.source_state_id == _vertex_tissue_state_fingerprint(source_state)
        )
        & source_evaluation.valid
    )
    capacity_valid = jnp.asarray(
        source.plan.dimension == target_plan.dimension
        and source.plan.vertex_capacity == target_plan.vertex_capacity
        and source.plan.edge_capacity == target_plan.edge_capacity
        and source.plan.face_capacity == target_plan.face_capacity
        and source.plan.cell_capacity == target_plan.cell_capacity
        and source.plan.field_count == target_plan.field_count
    )
    kind_host, lineage_host = _kind_and_lineage_valid(
        source.plan, target_plan, event.kind
    )
    kind_valid = jnp.asarray(kind_host)
    lineage_valid = jnp.asarray(lineage_host)
    lineage_transfer_valid = jnp.asarray(
        _lineage_transfer_valid(
            source.plan,
            target_plan,
            event.kind,
            event.cell_transfer,
            event.conservation_tolerance,
        )
    )
    transfer = event.cell_transfer
    source_active = source.plan.cell_active
    target_active = target_plan.cell_active
    column_sum = jnp.sum(transfer, axis=0)
    transfer_valid = (
        jnp.all(jnp.isfinite(transfer))
        & jnp.all(transfer >= 0.0)
        & jnp.all(
            jnp.where(source_active, jnp.abs(column_sum - 1.0), column_sum)
            <= event.conservation_tolerance
        )
        & jnp.all(
            jnp.where(target_active[:, None], 0.0, jnp.abs(transfer))
            <= event.conservation_tolerance
        )
    )
    expected_fields = oe.contract("ts,sf->tf", transfer, source_state.cell_fields)
    mapping_valid = (
        jnp.all(
            jnp.abs(candidate.state.cell_fields - expected_fields)
            <= event.conservation_tolerance
        )
        & (candidate.state.time == source_state.time)
        & jnp.all(candidate.state.positions == event.target_positions)
    )
    source_total = jnp.sum(
        jnp.where(source_active[:, None], source_state.cell_fields, 0.0), axis=0
    )
    target_total = jnp.sum(
        jnp.where(target_active[:, None], candidate.state.cell_fields, 0.0), axis=0
    )
    conservation_defect = jnp.sum(jnp.abs(target_total - source_total))
    conservation_valid = conservation_defect <= event.conservation_tolerance
    manifold = target.manifold
    orientation_valid = target.orientation_valid
    quality_valid = target.quality_valid
    finite = target.finite & jnp.isfinite(conservation_defect)
    passed = (
        fresh
        & source_state_valid
        & mapping_valid
        & capacity_valid
        & kind_valid
        & manifold
        & orientation_valid
        & quality_valid
        & transfer_valid
        & conservation_valid
        & lineage_valid
        & lineage_transfer_valid
        & finite
    )
    status = jnp.asarray(int(VertexTissueStatus.SUCCESS), dtype=jnp.int32)
    status = jnp.where(
        ~lineage_valid | ~lineage_transfer_valid,
        int(VertexTissueStatus.LINEAGE_FAILURE),
        status,
    )
    status = jnp.where(
        ~transfer_valid | ~conservation_valid | ~mapping_valid,
        int(VertexTissueStatus.CONSERVATION_FAILURE),
        status,
    )
    status = jnp.where(
        ~orientation_valid,
        int(VertexTissueStatus.ORIENTATION_FAILURE),
        status,
    )
    status = jnp.where(~quality_valid, int(VertexTissueStatus.QUALITY_FAILURE), status)
    status = jnp.where(~manifold, int(VertexTissueStatus.NONMANIFOLD), status)
    status = jnp.where(~kind_valid, int(VertexTissueStatus.INVALID_EVENT), status)
    status = jnp.where(~capacity_valid, int(VertexTissueStatus.CAPACITY_EXCEEDED), status)
    status = jnp.where(
        ~fresh | ~source_state_valid,
        int(VertexTissueStatus.STALE_EPOCH),
        status,
    )
    status = jnp.where(~finite, int(VertexTissueStatus.NONFINITE), status)
    return VertexTissueTopologyEvaluation(
        target,
        conservation_defect,
        fresh,
        source_state_valid,
        mapping_valid,
        capacity_valid,
        kind_valid,
        manifold,
        orientation_valid,
        quality_valid,
        transfer_valid,
        conservation_valid,
        lineage_transfer_valid,
        lineage_valid,
        finite,
        passed,
        status,
        candidate.candidate_id,
    )


class VertexTissueTopologyResult(StrictModule, NonTrainableState):
    """Atomic selected epoch/state with commit or rollback evidence."""

    prepared: PreparedVertexTissue
    state: VertexTissueState
    evaluation: VertexTissueTopologyEvaluation
    committed: Array
    status: Array
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        prepared: PreparedVertexTissue,
        state: VertexTissueState,
        evaluation: VertexTissueTopologyEvaluation,
        committed: ArrayLike,
        status: ArrayLike,
        result_id: str,
        /,
    ):
        if not isinstance(prepared, PreparedVertexTissue):
            raise TypeError("prepared must be a PreparedVertexTissue.")
        if not isinstance(state, VertexTissueState):
            raise TypeError("state must be a VertexTissueState.")
        if not isinstance(evaluation, VertexTissueTopologyEvaluation):
            raise TypeError("evaluation must be a VertexTissueTopologyEvaluation.")
        _validate_state(prepared, state)
        committed_ = jnp.asarray(committed)
        status_ = jnp.asarray(status, dtype=jnp.int32)
        if committed_.shape != () or committed_.dtype.kind != "b":
            raise ValueError("committed must be a scalar boolean.")
        if status_.shape != ():
            raise ValueError("status must be scalar.")
        if (
            not isinstance(result_id, str)
            or not result_id
            or result_id != result_id.strip()
        ):
            raise ValueError("result_id must be a nonempty canonical identifier.")
        self.prepared = prepared
        self.state = state
        self.evaluation = evaluation
        self.committed = committed_
        self.status = status_
        self.result_id = result_id


def _topology_result(
    source: PreparedVertexTissue,
    source_state: VertexTissueState,
    candidate: VertexTissueTopologyCandidate,
    evaluation: VertexTissueTopologyEvaluation,
    commit: bool,
    /,
) -> VertexTissueTopologyResult:
    if not isinstance(source, PreparedVertexTissue):
        raise TypeError("source must be a PreparedVertexTissue.")
    _validate_state(source, source_state)
    if not isinstance(candidate, VertexTissueTopologyCandidate):
        raise TypeError("candidate must be a VertexTissueTopologyCandidate.")
    if not isinstance(evaluation, VertexTissueTopologyEvaluation):
        raise TypeError("evaluation must be a VertexTissueTopologyEvaluation.")
    if evaluation.candidate_id != candidate.candidate_id:
        raise ValueError("Topology evaluation belongs to a different candidate.")
    source_matches = (
        candidate.event.source_prepared_id == source.prepared_id
        and candidate.source_state_id == _vertex_tissue_state_fingerprint(source_state)
    )
    evidence_passed = bool(np.asarray(evaluation.passed))
    accepted = evidence_passed and source_matches and commit
    prepared = candidate.prepared if accepted else source
    state = candidate.state if accepted else source_state
    if not source_matches:
        status = int(VertexTissueStatus.STALE_EPOCH)
    elif accepted:
        status = int(VertexTissueStatus.SUCCESS)
    elif evidence_passed:
        status = int(VertexTissueStatus.ROLLED_BACK)
    else:
        status = int(np.asarray(evaluation.status))
    result_id = canonical_fingerprint(
        {
            "kind": "vertex-tissue-topology-result",
            "source": source.prepared_id,
            "candidate": candidate.candidate_id,
            "selected": prepared.prepared_id,
            "committed": accepted,
            "status": status,
        }
    )
    return VertexTissueTopologyResult(
        prepared,
        state,
        evaluation,
        jnp.asarray(accepted),
        jnp.asarray(status, dtype=jnp.int32),
        result_id,
    )


def commit_vertex_tissue_topology(
    source: PreparedVertexTissue,
    source_state: VertexTissueState,
    candidate: VertexTissueTopologyCandidate,
    evaluation: VertexTissueTopologyEvaluation,
    /,
) -> VertexTissueTopologyResult:
    """Commit a passing candidate or return the source epoch and state unchanged."""

    return _topology_result(source, source_state, candidate, evaluation, True)


def rollback_vertex_tissue_topology(
    source: PreparedVertexTissue,
    source_state: VertexTissueState,
    candidate: VertexTissueTopologyCandidate,
    evaluation: VertexTissueTopologyEvaluation,
    /,
) -> VertexTissueTopologyResult:
    """Explicitly reject a candidate while retaining its complete evaluation."""

    return _topology_result(source, source_state, candidate, evaluation, False)


__all__ = [
    "PreparedVertexTissue",
    "PreparedVertexTissueDynamics",
    "VertexTissueDimension",
    "VertexTissueDynamicsPlan",
    "VertexTissueEvaluation",
    "VertexTissueEventKind",
    "VertexTissueParticleCoupling",
    "VertexTissuePlan",
    "VertexTissueState",
    "VertexTissueStatus",
    "VertexTissueStepResult",
    "VertexTissueTopologyCandidate",
    "VertexTissueTopologyEvaluation",
    "VertexTissueTopologyEvent",
    "VertexTissueTopologyResult",
    "commit_vertex_tissue_topology",
    "couple_vertex_tissue_particles",
    "evaluate_vertex_tissue",
    "evaluate_vertex_tissue_topology",
    "polygonal_vertex_tissue_plan",
    "polyhedral_vertex_tissue_plan",
    "propose_vertex_tissue_topology",
    "rollback_vertex_tissue_topology",
    "step_vertex_tissue_overdamped",
    "vertex_tissue_potential_energy",
]
