#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from itertools import combinations
from typing import final, NoReturn

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ...discretization._coordinate_enclosure import coordinate_polynomials, evaluate
from ...discretization._exact_algebraic_geometry import rational_coefficient
from ...discretization._reference_cell import reference_cell_topology
from ...discretization._simplicial_locator import (
    AbstractCellLocator,
    cell_map_vertices,
    CellLocationStatus,
)
from ...discretization.fem._cell_map import PreparedFiniteElementCellMap
from ...typing import checked, Dim, Float64, PRNGKey
from .._capabilities import GeometryCapability
from .._certificate import exact_signed_distance_certificate, FieldCertificate
from .._contracts import CompiledGeometry, GeometryKernel, GeometryKind
from .._sampling import RejectionSamplingPlan
from ..design._schema import DesignState, ParameterSchema


class _SupportFaceDim(Dim):
    """Faces in one dimension-homogeneous boundary projection workset."""


class _SupportAmbientDim(Dim):
    """Coordinates in the declared mesh frame."""


class _SupportSpanDim(Dim, minimum=0):
    """Affine span dimension, including vertex projections."""


@final
class _FaceProjection(StrictModule, NonTrainableState):
    __strict_contract__ = True
    origins: Float64[_SupportFaceDim, _SupportAmbientDim]
    edges: Float64[_SupportFaceDim, _SupportAmbientDim, _SupportSpanDim]
    duals: Float64[_SupportFaceDim, _SupportSpanDim, _SupportAmbientDim]
    rank: int = eqx.field(static=True)


@final
class SimplicialSupportKernel(GeometryKernel, NonTrainableState):
    """Query signed distance and containment for a fixed affine simplicial partition.

    Every subface of the exterior boundary participates in the nearest-point
    reduction. Unlike a minimum of per-cell fields, internal facets cannot
    become spurious boundary zeros. Geometry parameter derivatives require a
    rebuilt support; point derivatives remain JAX-native.
    """

    locator: AbstractCellLocator
    projections: tuple[_FaceProjection, ...]
    box: Array
    dimension: int = eqx.field(static=True)
    source_id: str = eqx.field(static=True)

    @property
    def ambient_dimension(self) -> int:
        return self.dimension

    @property
    def intrinsic_dimension(self) -> int:
        return self.dimension

    @property
    def kind(self) -> GeometryKind:
        return GeometryKind.REGION

    @property
    def capabilities(self) -> frozenset[GeometryCapability]:
        return frozenset(
            (GeometryCapability.REGION_QUERY, GeometryCapability.SIGNED_DISTANCE)
        )

    @property
    def field_certificate(self) -> FieldCertificate:
        return replace(
            exact_signed_distance_certificate(smooth=False),
            parameter_differentiable=False,
            topology_identity=self.source_id,
            provenance=("affine-simplex-exterior-face-projections",),
        )

    def _flat_points(self, points: Array, /) -> Array:
        if points.ndim < 1 or points.shape[-1] != self.dimension:
            raise ValueError(
                f"SimplicialSupportKernel.points must use mesh coordinate dimension {self.dimension}; got shape {points.shape}"
            )
        return points.reshape((-1, self.dimension))

    def _nearest_squared(self, point: Array, /) -> Array:
        minimum = jnp.asarray(jnp.inf, dtype=jnp.float64)
        # Worksets are heterogeneous in affine rank; this is a static loop,
        # while the potentially large face axes use bounded device loops.
        for workset in self.projections:

            def reduce_face(
                index: Array,
                current: Array,
                projection_workset: _FaceProjection = workset,
            ) -> Array:
                delta = point - projection_workset.origins[index]
                local = projection_workset.duals[index] @ delta
                barycentric = jnp.concatenate(
                    (
                        (1.0 - jnp.sum(local))[None],
                        local,
                    )
                )
                valid = jnp.all(barycentric >= -64.0 * jnp.finfo(jnp.float64).eps)
                projection = (
                    projection_workset.origins[index]
                    + projection_workset.edges[index] @ local
                )
                distance = jnp.sum((point - projection) ** 2)
                return jnp.minimum(current, jnp.where(valid, distance, jnp.inf))

            minimum = jax.lax.fori_loop(0, workset.origins.shape[0], reduce_face, minimum)
        return minimum

    def contains(self, state: DesignState, points: Array, /) -> Array:
        del state
        flat = self._flat_points(points)
        result = self.locator.locate(flat)
        references = result.candidate_reference
        barycentric = jnp.concatenate(
            (
                (1.0 - jnp.sum(references, axis=-1))[..., None],
                references,
            ),
            axis=-1,
        )
        contained = jnp.any(
            (result.candidate_cells >= 0)
            & jnp.all(barycentric >= -64.0 * jnp.finfo(jnp.float64).eps, axis=-1),
            axis=-1,
        )
        resolved = (
            (result.status == int(CellLocationStatus.LOCATED))
            | (result.status == int(CellLocationStatus.OUTSIDE))
            | (result.status == int(CellLocationStatus.NONFINITE))
        )
        inside = eqx.error_if(
            contained,
            jnp.any(~resolved),
            "Simplicial support query exhausted its certified point-location route.",
        )
        return inside.reshape(points.shape[:-1])

    def boundary_field(self, state: DesignState, points: Array, /) -> Array:
        flat = self._flat_points(points)
        squared = jax.lax.map(self._nearest_squared, flat).reshape(points.shape[:-1])
        positive = squared > 0.0
        distance = jnp.where(positive, jnp.sqrt(jnp.where(positive, squared, 1.0)), 0.0)
        signed = jnp.where(self.contains(state, points), -distance, distance)
        return jnp.where(jnp.all(jnp.isfinite(points), axis=-1), signed, jnp.nan)

    def bounds(self, state: DesignState, /) -> Array:
        del state
        return self.box

    def boundary_normal(self, state: DesignState, points: Array, /) -> NoReturn:
        del state, points
        raise NotImplementedError(
            "SimplicialSupportKernel.boundary_normal must use a provider of certified unique normals; got support-query geometry"
        )

    def measure(self, state: DesignState, /) -> NoReturn:
        del state
        raise NotImplementedError(
            "SimplicialSupportKernel.measure must use mesh integration; got support-query geometry without a certified measure"
        )

    def boundary_measure(self, state: DesignState, /) -> NoReturn:
        del state
        raise NotImplementedError(
            "SimplicialSupportKernel.boundary_measure must use boundary integration; got support-query geometry without a certified measure"
        )

    def sample_interior(
        self,
        state: DesignState,
        num_points: int,
        /,
        *,
        key: PRNGKey,
        plan: RejectionSamplingPlan | None = None,
    ) -> NoReturn:
        del state, num_points, key, plan
        raise NotImplementedError(
            "support_region.sample_interior must use mesh quadrature; got a geometry sampling request"
        )

    def sample_boundary(
        self, state: DesignState, num_points: int, /, *, key: PRNGKey
    ) -> NoReturn:
        del state, num_points, key
        raise NotImplementedError(
            "support_region.sample_boundary must use mesh boundary quadrature; got a geometry sampling request"
        )

    def boundary_atlas(self, state: DesignState, /) -> NoReturn:
        del state
        raise NotImplementedError(
            "SimplicialSupportKernel.boundary_atlas must use a boundary-atlas provider; got support-query geometry"
        )


def _boundary_faces(cells: np.ndarray, cell_kind: str, /) -> np.ndarray:
    reference = reference_cell_topology(cell_kind)
    face_indices = np.asarray(reference.entities[-2], dtype=np.int32)
    faces = cells[:, face_indices].reshape((-1, reference.dimension))
    keys = np.sort(faces, axis=1)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    if np.any(counts > 2):
        raise ValueError(
            "SimplicialSupport.facets must be manifold; got more than two incident cells"
        )
    boundary = keys[counts[inverse] == 1]
    if boundary.shape[0] == 0:
        raise ValueError(
            "SimplicialSupport.boundary must contain exterior facets for a bounded full-dimensional region; got no exterior facets"
        )
    return boundary


def _affine_support_coordinates(
    cell_map: PreparedFiniteElementCellMap,
    source_coordinates: np.ndarray,
    cells: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> np.ndarray:
    """Prove affine maps and realize their target corners, retaining source DOFs."""
    topology = reference_cell_topology(cell_map.coordinate_element.cell_kind)
    corners = tuple(
        tuple(Fraction(value) for value in point) for point in topology.vertices
    )
    exact_points: dict[int, tuple[Fraction, ...]] = {}
    routes = np.asarray(cell_map.coordinate_dofs, dtype=np.int32)
    source_bank = cell_map.source_coordinates(source_coordinates, budget=budget)
    for row in range(cells.shape[0]):
        polynomials = coordinate_polynomials(
            cell_map.coordinate_element,
            tuple(source_bank[routes[row, column]] for column in range(routes.shape[1])),
            budget=budget,
        )
        if (
            polynomials is None
            or len(polynomials) != topology.dimension
            or any(sum(index) > 1 for polynomial in polynomials for index in polynomial)
        ):
            raise ValueError(
                f"SimplicialSupport.coordinate_maps must be proven affine simplex maps; got an unrepresented or nonaffine map at cell {row}"
            )
        for column, corner in enumerate(corners):
            image = tuple(
                rational_coefficient(
                    evaluate(polynomial, corner, budget=budget),
                    role="SimplicialSupport.corner_image",
                )
                for polynomial in polynomials
            )
            identifier = int(cells[row, column])
            previous = exact_points.setdefault(identifier, image)
            if previous != image:
                raise ValueError(
                    f"SimplicialSupport.coordinate_maps must agree at shared mesh corners; got inconsistent corner {identifier}"
                )
    coordinates = np.zeros(
        (cell_map.mesh.coordinates.shape[0], topology.dimension), dtype=np.float64
    )
    for vertex, image in exact_points.items():
        coordinates[vertex] = tuple(float(value) for value in image)
    if not np.all(np.isfinite(coordinates)):
        raise ValueError(
            "SimplicialSupport.coordinates must have finite target corners; got nonfinite coordinates"
        )
    return coordinates


def _projection_worksets(
    coordinates: np.ndarray,
    faces: np.ndarray,
    dimension: int,
    /,
) -> tuple[_FaceProjection, ...]:
    worksets: list[_FaceProjection] = []
    for rank in range(dimension):
        rows = sorted(
            {
                tuple(int(vertex) for vertex in subface)
                for face in faces
                for subface in combinations(face, rank + 1)
            }
        )
        points = coordinates[np.asarray(rows, dtype=np.int32)]
        origins = points[:, 0]
        edges = np.swapaxes(points[:, 1:] - origins[:, None], -1, -2)
        duals = np.empty((len(rows), rank, dimension), dtype=np.float64)
        if rank:
            for index, matrix in enumerate(edges):
                # Immutable host geometry preparation; all projection actions
                # thereafter use the prepared maps, with no runtime inverse.
                dual, _, observed_rank, _ = np.linalg.lstsq(
                    matrix,
                    np.eye(dimension, dtype=np.float64),
                    rcond=None,
                )
                if observed_rank != rank:
                    raise ValueError(
                        f"SimplicialSupport.exterior_faces must have affine rank {rank}; got {observed_rank}"
                    )
                duals[index] = dual
        worksets.append(
            _FaceProjection(
                jnp.asarray(origins, dtype=jnp.float64),
                jnp.asarray(edges, dtype=jnp.float64),
                jnp.asarray(duals, dtype=jnp.float64),
                rank,
            )
        )
    return tuple(worksets)


@checked
def compile_simplicial_support(
    locator: AbstractCellLocator,
    support_id: str,
    /,
    *,
    maximum_face_entries: int = 1 << 20,
    maximum_retained_bytes: int = 256 << 20,
    budget: CoordinateEnclosureBudget,
) -> CompiledGeometry:
    """Compile affine simplex support in any dimension from its source map.

    Coordinate source expressions evaluated in exact arithmetic prove affinity
    and supply target-reference corner images. Retained coefficient routes are
    distinct from target corners; mesh vertex identities define shared facets.
    Non-affine or unrepresented maps require mapped-support geometry.
    """
    if not isinstance(support_id, str):
        raise TypeError(
            f"compile_simplicial_support.support_id must be a string; got {type(support_id).__name__}"
        )
    if not support_id:
        raise ValueError(
            f"compile_simplicial_support.support_id must be nonempty; got {support_id!r}"
        )
    for limit in (maximum_face_entries, maximum_retained_bytes):
        if not isinstance(limit, int) or isinstance(limit, bool):
            raise TypeError(
                f"compile_simplicial_support.budgets must be integers; got {type(limit).__name__}"
            )
        if limit < 1:
            raise ValueError(
                f"compile_simplicial_support.budgets must be positive; got {limit!r}"
            )
    cell_map = locator.cell_map
    if not isinstance(cell_map, PreparedFiniteElementCellMap):
        raise TypeError(
            f"compile_simplicial_support.cell_map must be a PreparedFiniteElementCellMap; got {type(cell_map).__name__}"
        )
    dimension = locator.cell_map.reference_dimension
    if locator.cell_map.ambient_dimension != dimension:
        raise ValueError(
            f"compile_simplicial_support.cell_map must be full-dimensional; got ambient dimension {locator.cell_map.ambient_dimension} and reference dimension {dimension}"
        )
    cells = cell_map_vertices(cell_map)
    if cells.shape[1:] != (dimension + 1,):
        raise ValueError(
            f"compile_simplicial_support.cells must have simplex vertex shape ({dimension + 1},); got {cells.shape[1:]}"
        )
    faces = _boundary_faces(cells, cell_map.coordinate_element.cell_kind)
    if maximum_face_entries < 1 or dimension > maximum_face_entries.bit_length():
        raise ValueError(
            f"compile_simplicial_support.face_entries must fit maximum_face_entries={maximum_face_entries}; got dimension {dimension} exceeding the budget"
        )
    if faces.shape[0] * (2**dimension - 1) > maximum_face_entries:
        raise ValueError(
            f"compile_simplicial_support.face_entries must fit maximum_face_entries={maximum_face_entries}; got too many exterior face subfaces"
        )
    # Admission bounds host/device retained projection maps before preparation.
    face_entries = faces.shape[0] * (2**dimension - 1)
    if face_entries * (2 * dimension**2 + dimension) * 8 > maximum_retained_bytes:
        raise ValueError(
            f"compile_simplicial_support.projection_maps must fit maximum_retained_bytes={maximum_retained_bytes}; got retained bytes above the budget"
        )
    coordinates = _affine_support_coordinates(
        cell_map, np.asarray(locator.coordinates, dtype=np.float64), cells, budget=budget
    )
    used = coordinates[np.unique(cells)]
    kernel = SimplicialSupportKernel(
        locator,
        _projection_worksets(coordinates, faces, dimension),
        jnp.asarray(np.stack((used.min(axis=0), used.max(axis=0))), dtype=jnp.float64),
        dimension,
        support_id,
    )
    state = DesignState(ParameterSchema(()), ())
    return CompiledGeometry(kernel, state)
