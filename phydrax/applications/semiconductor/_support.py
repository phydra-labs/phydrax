#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Positive two-point transport metrics and exact native mesh selections."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations
from math import factorial

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...meshing import (
    CellMeshingResult,
    MeshingEntityKind,
    MeshingScope,
    MeshPatch,
)
from ...units import derived_unit, METER, ONE, UnitDefinition
from ._quantities import _si, SQUARE_METER


def _dual_widths(axis):
    delta = np.diff(axis)
    return np.concatenate((delta[:1] / 2, (delta[:-1] + delta[1:]) / 2, delta[-1:] / 2))


def _transverse(value, unit, dimension):
    reference = (
        ONE
        if dimension == 3
        else derived_unit(f"m{3 - dimension}", ((METER, 3 - dimension),))
    )
    measure = np.asarray(_si(value, reference if unit is None else unit, reference))
    if measure.shape != () or not np.isfinite(measure) or measure <= 0:
        raise ValueError(
            "transverse_measure must be one finite positive physical measure."
        )
    return float(measure)


def _simplex_metric(points):
    """Affine simplex gradients without a host linear-system solver."""
    dimension = points.shape[1]
    edges = points[1:] - points[0]
    if dimension == 1:
        determinant = edges[0, 0]
        reciprocal = np.asarray([[1.0 / determinant]]) if determinant != 0 else None
    elif dimension == 2:
        a, b = edges
        determinant = a[0] * b[1] - a[1] * b[0]
        reciprocal = (
            np.asarray([[b[1], -b[0]], [-a[1], a[0]]]) / determinant
            if determinant != 0
            else None
        )
    else:
        a, b, c = edges
        determinant = float(np.sum(a * np.cross(b, c)))
        reciprocal = (
            np.stack((np.cross(b, c), np.cross(c, a), np.cross(a, b))) / determinant
            if determinant != 0
            else None
        )
    if reciprocal is None or not np.isfinite(determinant):
        raise ValueError("Transport requires nondegenerate affine simplices.")
    gradients = np.concatenate((-np.sum(reciprocal, axis=0, keepdims=True), reciprocal))
    return abs(determinant) / factorial(dimension), gradients


class TransportSupport(StrictModule):
    """Node-centred conservative support in physical SI units.

    ``volumes`` are physical cubic metres, including extrusion in 1D/2D.
    ``transmissibility`` has SI metres after extrusion: multiplying it by
    diffusivity (m2/s) and density (1/m3) gives integrated number flux (1/s).
    An oriented flux is scattered with minus at tail and plus at head.

    Simplex supports use lumped P1 volumes and assembled negative off-diagonal
    stiffness weights. Only strictly positive two-point weights are admitted;
    non-Delaunay/obtuse metrics are never clipped into artificial diffusion.
    """

    positions: Array
    tail: Array
    head: Array
    volumes: Array
    transmissibility: Array
    boundary_mask: Array
    node_ids: Array
    entity_ids: tuple[Array, ...]
    entity_vertices: tuple[Array, ...]
    entity_set_ids: tuple[str, ...] = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    source_topology_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)

    def __init__(
        self,
        positions,
        tail,
        head,
        volumes,
        transmissibility,
        boundary_mask,
        /,
        *,
        node_ids=None,
        source_id=None,
        source_revision="0",
        source_topology_id="",
        entity_ids=(),
        entity_vertices=(),
        entity_set_ids=(),
    ):
        points = np.asarray(positions, dtype=float)
        tails, heads = np.asarray(tail), np.asarray(head)
        weights, measure = (
            np.asarray(transmissibility, dtype=float),
            np.asarray(volumes, dtype=float),
        )
        boundary = np.asarray(boundary_mask)
        if (
            points.ndim != 2
            or points.shape[1] not in (1, 2, 3)
            or points.shape[0] < 2
            or not np.all(np.isfinite(points))
        ):
            raise ValueError(
                "positions must be finite (N >= 2, d in 1,2,3) SI coordinates."
            )
        count = len(points)
        if (
            tails.ndim != 1
            or tails.size == 0
            or heads.shape != tails.shape
            or tails.dtype.kind not in "iu"
            or heads.dtype.kind not in "iu"
        ):
            raise ValueError(
                "tail and head must be equally sized nonempty integer vectors."
            )
        if np.any(tails < 0) or np.any(heads >= count) or np.any(tails >= heads):
            raise ValueError("Edges require 0 <= tail < head < N.")
        edges = np.stack((tails, heads), axis=1)
        if len(np.unique(edges, axis=0)) != len(edges):
            raise ValueError("Duplicate transport edges are not admitted.")
        if (
            weights.shape != tails.shape
            or not np.all(np.isfinite(weights))
            or np.any(weights <= 0)
        ):
            raise ValueError(
                "Two-point transmissibility must be finite and strictly positive."
            )
        if (
            measure.shape != (count,)
            or not np.all(np.isfinite(measure))
            or np.any(measure <= 0)
        ):
            raise ValueError("Every node must have a finite positive physical volume.")
        if boundary.shape != (count,) or boundary.dtype.kind != "b":
            raise ValueError("boundary_mask must be a boolean nodal vector.")
        if np.unique(edges).size != count:
            raise ValueError("Every transport node must participate in an edge.")
        identifiers = (
            np.arange(count, dtype=np.int64) if node_ids is None else np.asarray(node_ids)
        )
        if (
            identifiers.shape != (count,)
            or identifiers.dtype.kind not in "iu"
            or np.any(identifiers < 0)
            or len(np.unique(identifiers)) != count
        ):
            raise ValueError("node_ids must be unique nonnegative integer identifiers.")
        identity = canonical_fingerprint(
            {
                "kind": "semiconductor-transport-support",
                "positions": array_tree_fingerprint(points),
                "edges": array_tree_fingerprint(edges),
                "volumes": array_tree_fingerprint(measure),
                "weights": array_tree_fingerprint(weights),
                "node_ids": array_tree_fingerprint(identifiers),
                "boundary": array_tree_fingerprint(boundary),
                "source": source_id,
                "revision": source_revision,
            }
        )
        source = identity if source_id is None else str(source_id)
        if not source or not str(source_revision):
            raise ValueError("Mesh source identity and revision must be nonempty.")
        sets = tuple(entity_set_ids)
        ids, routes = tuple(entity_ids), tuple(entity_vertices)
        if not sets:
            sets = (
                canonical_fingerprint({"kind": "semiconductor-nodes", "source": source}),
            )
            ids, routes = (identifiers,), (np.arange(count)[:, None],)
        if len(sets) != len(ids) or len(sets) != len(routes):
            raise ValueError(
                "Native entity IDs, vertex routes and set identities must match."
            )
        for entity_identifiers, vertices in zip(ids, routes, strict=True):
            if np.asarray(vertices).ndim != 2 or np.asarray(vertices).shape[0] != len(
                entity_identifiers
            ):
                raise ValueError("Native entity routes must have one row per entity.")
            if np.any(np.asarray(vertices) < 0) or np.any(np.asarray(vertices) >= count):
                raise ValueError("Native entity routes must index support vertices.")
        self.positions, self.tail, self.head = (
            jnp.asarray(points),
            jnp.asarray(tails, dtype=jnp.int32),
            jnp.asarray(heads, dtype=jnp.int32),
        )
        self.volumes, self.transmissibility = jnp.asarray(measure), jnp.asarray(weights)
        self.boundary_mask, self.node_ids = (
            jnp.asarray(boundary),
            jnp.asarray(identifiers),
        )
        self.entity_ids = tuple(jnp.asarray(value) for value in ids)
        self.entity_vertices = tuple(
            jnp.asarray(value, dtype=jnp.int32) for value in routes
        )
        self.entity_set_ids = sets
        self.source_id, self.source_revision = source, str(source_revision)
        self.source_topology_id = str(source_topology_id)
        self.support_id = identity

    @classmethod
    def interval(
        cls,
        coordinates: ArrayLike,
        /,
        *,
        area: ArrayLike = 1.0,
        length_unit: UnitDefinition = METER,
        area_unit: UnitDefinition = SQUARE_METER,
    ):
        return cls.tensor_grid(
            (coordinates,),
            transverse_measure=area,
            length_unit=length_unit,
            transverse_unit=area_unit,
        )

    @classmethod
    def tensor_grid(
        cls,
        axes: Sequence[ArrayLike],
        /,
        *,
        transverse_measure=1.0,
        length_unit: UnitDefinition = METER,
        transverse_unit: UnitDefinition | None = None,
    ):
        axes_si = tuple(
            np.asarray(_si(axis, length_unit, METER), dtype=float) for axis in axes
        )
        dimension = len(axes_si)
        if dimension not in (1, 2, 3):
            raise ValueError("Tensor supports require one, two or three physical axes.")
        for axis in axes_si:
            if (
                axis.ndim != 1
                or len(axis) < 2
                or not np.all(np.isfinite(axis))
                or np.any(np.diff(axis) <= 0)
            ):
                raise ValueError(
                    "Each tensor axis must be finite and strictly increasing with at least two nodes."
                )
        transverse = _transverse(transverse_measure, transverse_unit, dimension)
        shape = tuple(len(axis) for axis in axes_si)
        indices = np.arange(np.prod(shape)).reshape(shape)
        widths = tuple(_dual_widths(axis) for axis in axes_si)
        positions = np.stack(np.meshgrid(*axes_si, indexing="ij"), axis=-1).reshape(
            -1, dimension
        )
        measures = np.full(shape, transverse)
        for axis, width in enumerate(widths):
            reshape = [1] * dimension
            reshape[axis] = len(width)
            measures *= width.reshape(reshape)
        tails, heads, weights = [], [], []
        boundary = np.zeros(shape, dtype=bool)
        for axis in range(dimension):
            lower, upper = [slice(None)] * dimension, [slice(None)] * dimension
            lower[axis], upper[axis] = slice(None, -1), slice(1, None)
            tails.append(indices[tuple(lower)].ravel())
            heads.append(indices[tuple(upper)].ravel())
            edge_shape = list(shape)
            edge_shape[axis] -= 1
            edge_weights = np.full(edge_shape, transverse)
            for other in range(dimension):
                factor = 1.0 / np.diff(axes_si[axis]) if other == axis else widths[other]
                reshape = [1] * dimension
                reshape[other] = len(factor)
                edge_weights *= factor.reshape(reshape)
            weights.append(edge_weights.ravel())
            lower[axis], upper[axis] = 0, -1
            boundary[tuple(lower)] = True
            boundary[tuple(upper)] = True
        return cls(
            positions,
            np.concatenate(tails),
            np.concatenate(heads),
            measures.ravel(),
            np.concatenate(weights),
            boundary.ravel(),
        )

    @classmethod
    def from_meshing(
        cls,
        result: CellMeshingResult,
        /,
        *,
        transverse_measure=1.0,
        transverse_unit: UnitDefinition | None = None,
    ):
        if not isinstance(result, CellMeshingResult):
            raise TypeError("from_meshing requires an audited native CellMeshingResult.")
        mesh = result.mesh
        dimension = mesh.topological_dimension
        if dimension not in (1, 2, 3) or mesh.ambient_dimension != dimension:
            raise ValueError(
                "Transport requires full-dimensional 1D/2D/3D affine simplices."
            )
        if (
            result.coordinate_contract.length_coordinate_kind != "physical"
            or result.coordinate_contract.coordinate_system != "cartesian"
        ):
            raise ValueError("Transport requires physical Cartesian mesh coordinates.")
        expected_kind = {1: "interval", 2: "triangle", 3: "tetrahedron"}[dimension]
        if any(block.cell_kind != expected_kind for block in mesh.blocks):
            raise ValueError(
                "Native transport currently admits affine simplicial blocks only."
            )
        elements, geometry_routes, coordinates = result.geometry.resolve(mesh)
        if np.asarray(coordinates).shape != np.asarray(mesh.coordinates).shape:
            raise ValueError(
                "High-order geometry is not an affine two-point transport metric."
            )
        for block, element, route in zip(
            mesh.blocks, elements, geometry_routes, strict=True
        ):
            if element.local_dof_count != dimension + 1 or not np.array_equal(
                route, block.vertices
            ):
                raise ValueError(
                    "Curved or non-vertex geometry is unsupported for two-point transport."
                )
        points = np.asarray(
            _si(coordinates, result.coordinate_contract.length_unit, METER)
        )
        transverse = _transverse(transverse_measure, transverse_unit, dimension)
        cells = np.concatenate(tuple(np.asarray(block.vertices) for block in mesh.blocks))
        cell_ids = np.concatenate(
            tuple(np.asarray(block.global_ids) for block in mesh.blocks)
        )
        volumes = np.zeros(len(points))
        edge_weights = {}
        for cell in cells:
            volume, gradients = _simplex_metric(points[cell])
            if not np.isfinite(volume) or volume <= 0:
                raise ValueError("Simplex volumes must be finite and positive.")
            np.add.at(volumes, cell, volume * transverse / (dimension + 1))
            for i, j in combinations(range(dimension + 1), 2):
                key = tuple(sorted((int(cell[i]), int(cell[j]))))
                weight = -volume * transverse * float(np.sum(gradients[i] * gradients[j]))
                edge_weights[key] = edge_weights.get(key, 0.0) + weight
        edges = np.asarray(sorted(edge_weights), dtype=np.int32)
        weights = np.asarray([edge_weights[tuple(edge)] for edge in edges])
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
            raise ValueError(
                "Native simplex mesh has nonpositive two-point metrics; an admissible positive mesh is required."
            )
        entity_routes = [np.arange(len(points))[:, None]]
        if dimension >= 2:
            entity_routes.append(np.asarray(mesh.connectivity.edges))
        if dimension == 3:
            entity_routes.append(np.asarray(mesh.connectivity.faces))
        cell_lookup = {
            int(identifier): cell
            for identifier, cell in zip(cell_ids, cells, strict=True)
        }
        entity_routes.append(
            np.asarray(
                [
                    cell_lookup[int(identifier)]
                    for identifier in np.asarray(mesh.entity_set(dimension).entity_ids)
                ]
            )
        )
        return cls(
            points,
            edges[:, 0],
            edges[:, 1],
            volumes,
            weights,
            mesh.connectivity.boundary_vertices,
            node_ids=mesh.vertex_global_ids,
            source_id=mesh.mesh_id,
            source_revision=mesh.numeric_version,
            source_topology_id=mesh.topology_id,
            entity_ids=tuple(mesh.entity_set(d).entity_ids for d in range(dimension + 1)),
            entity_vertices=tuple(entity_routes),
            entity_set_ids=tuple(
                mesh.entity_set(d).entity_set_id for d in range(dimension + 1)
            ),
        )

    def node_scope(self, node_ids: ArrayLike | None = None, /) -> MeshingScope:
        """Create a native scope from persistent node IDs, never array positions."""
        scope = MeshingScope(
            self.source_id,
            self.source_revision,
            MeshingEntityKind.MESH,
            0,
            self.entity_set_ids[0],
            self.node_ids if node_ids is None else node_ids,
        )
        self.resolve_scope(scope)
        return scope

    def boundary_patch(
        self, name: str, /, *, axis: int = 0, side: str = "lower"
    ) -> MeshPatch:
        if axis not in range(self.positions.shape[1]) or side not in ("lower", "upper"):
            raise ValueError("Boundary patch requires a valid axis and lower/upper side.")
        positions = np.asarray(self.positions)[:, axis]
        coordinate = np.min(positions) if side == "lower" else np.max(positions)
        mask = np.asarray(self.boundary_mask) & (positions == coordinate)
        return MeshPatch(name, self.node_scope(np.asarray(self.node_ids)[mask]))

    def resolve_scope(self, scope: MeshingScope, /) -> np.ndarray:
        """Host-only exact binding resolution to a nodal mask.

        Facet/cell scopes include every incident vertex. Competing cell-region
        assignments at an interface therefore require an explicit nodal zone;
        no arbitrary last-writer material ownership is inferred.
        """
        if not isinstance(scope, MeshingScope):
            raise TypeError("Selections must use native MeshingScope bindings.")
        if (
            scope.source_id != self.source_id
            or scope.source_revision != self.source_revision
        ):
            raise ValueError("Stale or foreign mesh scope source/revision.")
        dimension = scope.entity_dimension
        if (
            scope.entity_kind != MeshingEntityKind.MESH
            or dimension >= len(self.entity_ids)
            or scope.entity_set_id != self.entity_set_ids[dimension]
        ):
            raise ValueError("Scope must bind an exact native mesh entity set.")
        ids = np.asarray(self.entity_ids[dimension])
        selected = np.asarray(scope.entity_ids)
        lookup = {int(identifier): i for i, identifier in enumerate(ids)}
        if any(int(identifier) not in lookup for identifier in selected):
            raise ValueError("Scope includes unknown or inactive mesh entities.")
        rows = np.asarray([lookup[int(identifier)] for identifier in selected])
        mask = np.zeros(self.positions.shape[0], dtype=bool)
        mask[np.asarray(self.entity_vertices[dimension])[rows].ravel()] = True
        return mask


__all__ = ["TransportSupport"]
