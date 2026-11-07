#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Affine-consistent, coercive hybrid mimetic diffusion on admissible 3-D FV cells."""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ..._strict import StrictModule
from ...ein import contract
from ...linalg import (
    ArraySpace,
    FunctionLinearOperator,
    GMRES,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSystem,
    solve,
)
from ...typing import checked
from .._cell_complex import PolyhedralConnectivity
from ._diffusion_boundary import HybridDiffusionBoundary
from ._unstructured import UnstructuredFiniteVolumeDiscretization


def _tensor(value: ArrayLike, count: int) -> Array:
    result = jnp.asarray(value)
    result = result.astype(jnp.result_type(result, 1.0))
    if result.shape == ():
        return jnp.broadcast_to(result * jnp.eye(3), (count, 3, 3))
    if result.shape == (count,):
        return result[:, None, None] * jnp.eye(3)
    if result.shape == (3, 3):
        return jnp.broadcast_to(result, (count, 3, 3))
    if result.shape != (count, 3, 3):
        raise ValueError("Tensor must be scalar, (cells,), (3,3), or (cells,3,3).")
    return result


def _positive_tensor(value: Array) -> Array:
    """Sylvester certificate, without factoring or altering the physical tensor."""
    a, b, c = value[:, 0, 0], value[:, 0, 1], value[:, 0, 2]
    d, e, f = value[:, 1, 1], value[:, 1, 2], value[:, 2, 2]
    determinant = a * (d * f - e * e) - b * (b * f - c * e) + c * (b * e - c * d)
    scale = jnp.max(jnp.abs(value), axis=(1, 2))
    symmetric = jnp.all(
        jnp.abs(value - jnp.swapaxes(value, 1, 2))
        <= 32 * jnp.finfo(value.dtype).eps * scale[:, None, None]
    )
    return eqx.error_if(
        value,
        ~symmetric
        | ~jnp.all(jnp.isfinite(value))
        | jnp.any((a <= 0) | (a * d - b * b <= 0) | (determinant <= 0)),
        "Hybrid diffusion requires finite symmetric positive-definite cell tensors.",
    )


class _TraceFacets(NamedTuple):
    faces: np.ndarray
    centers: np.ndarray
    areas: np.ndarray
    measures: np.ndarray
    shares: np.ndarray


def _trace_facets(
    discretization: UnstructuredFiniteVolumeDiscretization, /
) -> _TraceFacets:
    """Planar FV faces stay whole; warped faces split into their FV sub-triangles.

    Polyhedral FV geometry fans a planar n-gon into at most n - 2 triangles and
    splits a warped n-gon into exactly n triangles about its vertex centroid.
    Cell volumes, centroids and the face area vector integrate those triangles,
    so each warped sub-triangle is one planar trace facet whose vector area and
    centroid carry its first moment exactly. Both incident cells share it with
    opposite orientation. ``shares`` are projected-area fractions of the parent
    face; their weighted facet centroid is the FV face center.
    """
    centers = np.asarray(discretization.face_centers)
    areas = np.asarray(discretization.area_vectors)
    measures = np.asarray(discretization.face_measures)
    count = measures.size
    whole = _TraceFacets(
        np.arange(count, dtype=np.int32),
        centers,
        areas,
        measures,
        np.ones(count, dtype=centers.dtype),
    )
    connectivity = discretization.connectivity
    mesh = discretization.mesh
    if not isinstance(connectivity, PolyhedralConnectivity) or mesh is None:
        return whole
    topology = discretization.face_topology_ids
    rows = np.arange(count) if topology is None else np.asarray(topology)
    offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
    arity = np.diff(offsets)[rows]
    triangles = np.count_nonzero(
        np.asarray(discretization.face_quadrature_weights) > 0, axis=1
    )
    split = triangles == arity
    split_faces = np.flatnonzero(split)
    if split_faces.size == 0:
        return whole
    sides = arity[split_faces][:, None]
    corner = np.arange(int(np.max(sides)))[None, :]
    loop = corner < sides
    starts = offsets[rows[split_faces]][:, None]
    vertices = np.asarray(connectivity.face_vertex_values)[
        np.where(loop, starts + corner, starts)
    ]
    following = np.take_along_axis(
        vertices, np.where(corner + 1 < sides, corner + 1, 0), axis=1
    )
    points = np.asarray(mesh.coordinates, dtype=centers.dtype)
    apex = np.sum(np.where(loop[..., None], points[vertices], 0), axis=1) / sides
    first = points[vertices] - apex[:, None, :]
    second = points[following] - apex[:, None, :]
    sub_areas = 0.5 * np.cross(first, second)
    sub_centers = apex[:, None, :] + (first + second) / 3
    loop_area = np.sum(np.where(loop[..., None], sub_areas, 0), axis=1)
    projected = np.where(
        loop,
        contract("sti,si->st", sub_areas, loop_area)
        / np.linalg.norm(loop_area, axis=-1)[:, None],
        0,
    )
    sub_shares = projected / np.sum(projected, axis=1, keepdims=True)
    signs = np.asarray(discretization.owner_signs)[split_faces].astype(centers.dtype)
    counts = np.where(split, arity, 1)
    faces = np.repeat(np.arange(count, dtype=np.int32), counts)
    local = np.arange(faces.size) - np.repeat(np.cumsum(counts) - counts, counts)
    member = split[faces]
    rank = (np.cumsum(split) - 1)[faces[member]]
    slot = local[member]
    facet_centers = centers[faces]
    facet_areas = areas[faces]
    facet_measures = measures[faces]
    shares = np.ones(faces.size, dtype=centers.dtype)
    facet_centers[member] = sub_centers[rank, slot]
    facet_areas[member] = signs[rank, None] * sub_areas[rank, slot]
    facet_measures[member] = np.linalg.norm(sub_areas[rank, slot], axis=-1)
    shares[member] = sub_shares[rank, slot]
    return _TraceFacets(faces, facet_centers, facet_areas, facet_measures, shares)


class HybridMimeticDiffusion(StrictModule):
    """Fixed polyhedral geometry, shared global trace potentials, no TPFA reduction.

    Hybrid trace unknowns live on trace facets: a planar FV face is one facet; a
    warped face contributes the planar sub-triangles about its vertex centroid
    that the FV geometry integrates for cell volumes, centroids and face area
    vectors. Cells must be strictly star-shaped about their FV centers over their
    facets, have positive measures, outward closure and exact first geometric
    moments. The local energy is V grad(u).K.grad(u) plus a positive
    normal-distance weighted penalty on deviations from affine facet traces. This
    is consistent for full rotated SPD tensors, including off-diagonal entries.

    Boundary laws stay FV-face indexed. A Dirichlet value is the uniform trace on
    every facet of its face; integrated Neumann rates and Robin conductances are
    split over facets by projected-area share. Face rates sum their facets, and
    ``face_potentials`` returns the projected-area mean trace of each face.
    """

    discretization: UnstructuredFiniteVolumeDiscretization
    cell_facets: Array
    valid: Array
    outward_areas: Array
    displacements: Array
    gradient_weights: Array
    defect: Array
    stabilization_weights: Array
    facet_faces: Array
    facet_centers: Array
    facet_measures: Array
    facet_shares: Array
    facet_owners: Array
    owner_slots: Array
    component_ids: Array
    component_count: int = eqx.field(static=True)
    stabilization: float = eqx.field(static=True)

    @checked
    def __init__(
        self,
        discretization: UnstructuredFiniteVolumeDiscretization,
        /,
        *,
        stabilization: float = 1.0,
    ) -> None:
        if discretization.cell_dimension != 3:
            raise ValueError("Hybrid mimetic diffusion requires three-dimensional cells.")
        if not np.isfinite(stabilization) or stabilization <= 0:
            raise ValueError("stabilization must be finite and strictly positive.")
        owner = np.asarray(discretization.owner_cells)
        neighbor = np.asarray(discretization.neighbor_cells)
        centers = np.asarray(discretization.cell_centers)
        volumes = np.asarray(discretization.cell_volumes)
        count = volumes.size
        facets = _trace_facets(discretization)
        facet_count = facets.faces.size
        facet_ids = np.arange(facet_count, dtype=np.int32)
        facet_owners = owner[facets.faces]
        facet_neighbors = neighbor[facets.faces]
        interior = facet_neighbors >= 0
        # Cell-local slots list incident facets in ascending facet order.
        entry_cells = np.concatenate((facet_owners, facet_neighbors[interior]))
        entry_facets = np.concatenate((facet_ids, facet_ids[interior]))
        entry_signs = np.concatenate(
            (np.ones(facet_count), -np.ones(np.count_nonzero(interior)))
        )
        order = np.lexsort((entry_facets, entry_cells))
        entry_cells = entry_cells[order]
        entry_facets = entry_facets[order]
        entry_signs = entry_signs[order]
        incident = np.bincount(entry_cells, minlength=count)
        slots = np.arange(entry_cells.size) - np.repeat(
            np.cumsum(incident) - incident, incident
        )
        width = int(np.max(incident, initial=0))
        cell_facets = np.zeros((count, width), dtype=np.int32)
        orientation = np.zeros((count, width))
        cell_facets[entry_cells, slots] = entry_facets
        orientation[entry_cells, slots] = entry_signs
        owned = entry_signs > 0
        owner_slots = np.zeros(facet_count, dtype=np.int32)
        owner_slots[entry_facets[owned]] = slots[owned]
        valid = orientation != 0
        measures = facets.measures[cell_facets]
        outward = facets.areas[cell_facets] * orientation[..., None]
        offsets = (facets.centers[cell_facets] - centers[:, None, :]) * valid[..., None]
        distance = contract("cfi,cfi->cf", offsets, outward) / measures
        if np.any(~np.isfinite(volumes)) or np.any(volumes <= 0):
            raise ValueError("Hybrid diffusion cell volumes must be positive and finite.")
        if np.any(~np.isfinite(distance)) or np.any(distance[valid] <= 0):
            raise ValueError("Cells must be strictly star-shaped about their FV centers.")
        weights = outward / volumes[:, None, None]
        moment = contract("cfi,cfj->cij", weights, offsets)
        tolerance = 512 * np.finfo(centers.dtype).eps
        if not np.allclose(moment, np.eye(3), rtol=tolerance, atol=tolerance):
            raise ValueError("Hybrid geometry fails the affine first-moment identity.")
        closure = np.sum(outward, axis=1)
        area_scale = np.sum(measures * valid, axis=1)
        if np.any(np.abs(closure) > tolerance * area_scale[:, None]):
            raise ValueError("Hybrid cell outward area vectors must close.")
        defect = np.eye(width)[None] * valid[:, :, None] - contract(
            "cfi,cgi->cfg", offsets, weights
        )
        # Connectivity components are also the nullspace components of the energy.
        joined = neighbor >= 0
        component_count, labels = connected_components(
            coo_matrix(
                (np.ones(np.count_nonzero(joined)), (owner[joined], neighbor[joined])),
                shape=(count, count),
            ),
            directed=False,
        )
        self.discretization = discretization
        self.cell_facets = jnp.asarray(cell_facets)
        self.valid = jnp.asarray(valid)
        self.outward_areas = jnp.asarray(outward)
        self.displacements = jnp.asarray(offsets)
        self.gradient_weights = jnp.asarray(weights)
        self.defect = jnp.asarray(defect)
        self.stabilization_weights = jnp.asarray(
            measures / np.where(valid, distance, 1.0) * valid
        )
        self.facet_faces = jnp.asarray(facets.faces)
        self.facet_centers = jnp.asarray(facets.centers)
        self.facet_measures = jnp.asarray(facets.measures)
        self.facet_shares = jnp.asarray(facets.shares)
        self.facet_owners = jnp.asarray(facet_owners)
        self.owner_slots = jnp.asarray(owner_slots)
        self.component_ids = jnp.asarray(labels.astype(np.int32))
        self.component_count = int(component_count)
        self.stabilization = float(stabilization)

    @property
    def cell_count(self) -> int:
        return self.discretization.cell_volumes.size

    @property
    def face_count(self) -> int:
        return self.discretization.face_measures.size

    @property
    def facet_count(self) -> int:
        return self.facet_faces.size

    def local_matrices(self, tensor: ArrayLike) -> Array:
        tensor = _positive_tensor(_tensor(tensor, self.cell_count))
        normal = self.outward_areas / self.facet_measures[self.cell_facets, None]
        normal_conductivity = contract("cfi,cij,cfj->cf", normal, tensor, normal)
        penalty = self.stabilization * self.stabilization_weights * normal_conductivity
        consistent = self.discretization.cell_volumes[:, None, None] * contract(
            "cfi,cij,cgj->cfg", self.gradient_weights, tensor, self.gradient_weights
        )
        return consistent + contract("cfg,cf,cfh->cgh", self.defect, penalty, self.defect)

    def gradient(self, cell_values: ArrayLike, facet_values: ArrayLike) -> Array:
        difference = (
            jnp.asarray(facet_values)[self.cell_facets]
            - jnp.asarray(cell_values)[:, None]
        )
        return contract("cfi,cf->ci", self.gradient_weights, difference)

    def local_fluxes(
        self,
        cell_values: ArrayLike,
        facet_values: ArrayLike,
        tensor: ArrayLike,
        *,
        body_force: ArrayLike | None = None,
    ) -> Array:
        tensor = _tensor(tensor, self.cell_count)
        difference = (
            jnp.asarray(facet_values)[self.cell_facets]
            - jnp.asarray(cell_values)[:, None]
        )
        result = -contract("cfg,cg->cf", self.local_matrices(tensor), difference)
        if body_force is not None:
            force = jnp.broadcast_to(jnp.asarray(body_force), (self.cell_count, 3))
            result = result + contract(
                "cfi,cij,cj->cf", self.outward_areas, tensor, force
            )
        return jnp.where(self.valid, result, 0.0)

    def owner_rates(self, local_fluxes: Array) -> Array:
        """Owner-outward FV face rates: the sum of each face's facet rates."""
        rates = local_fluxes[self.facet_owners, self.owner_slots]
        return (
            jnp.zeros(self.face_count, dtype=rates.dtype).at[self.facet_faces].add(rates)
        )

    def face_fluxes(
        self,
        cell_values: ArrayLike,
        facet_values: ArrayLike,
        tensor: ArrayLike,
        *,
        body_force: ArrayLike | None = None,
    ) -> Array:
        return self.owner_rates(
            self.local_fluxes(cell_values, facet_values, tensor, body_force=body_force)
        )

    def face_potentials(self, facet_values: ArrayLike) -> Array:
        """Projected-area mean trace of every FV face, exact for affine traces."""
        values = jnp.asarray(facet_values)
        return (
            jnp.zeros(self.face_count, dtype=values.dtype)
            .at[self.facet_faces]
            .add(self.facet_shares * values)
        )

    def continuity_residual(self, local_fluxes: Array) -> Array:
        return (
            jnp.zeros(self.facet_count, dtype=local_fluxes.dtype)
            .at[self.cell_facets]
            .add(jnp.where(self.valid, local_fluxes, 0.0))
        )

    def cell_divergence(self, face_rates: ArrayLike) -> Array:
        """Net outward integrated owner rates, without division by cell volume."""
        owner = self.discretization.owner_cells
        neighbor = self.discretization.neighbor_cells
        rates = jnp.asarray(face_rates)
        result = jnp.zeros(self.cell_count, dtype=rates.dtype).at[owner].add(rates)
        return result.at[jnp.maximum(neighbor, 0)].add(
            jnp.where(neighbor >= 0, -rates, 0.0)
        )

    def anchored_components(self, boundary: HybridDiffusionBoundary) -> Array:
        anchored = (boundary.kind == 1) | (
            (boundary.kind == 3) & (boundary.conductance > 0)
        )
        return (
            jnp.zeros(self.component_count, dtype=jnp.int32)
            .at[self.component_ids[self.discretization.owner_cells]]
            .max(anchored.astype(jnp.int32))
            > 0
        )

    def _impose_dirichlet(
        self, boundary: HybridDiffusionBoundary, facet_values: Array
    ) -> Array:
        faces = self.facet_faces
        return jnp.where(boundary.kind[faces] == 1, boundary.value[faces], facet_values)

    def _trace_residual(
        self,
        boundary: HybridDiffusionBoundary,
        facet_values: Array,
        outward_sum: Array,
    ) -> Array:
        """Facet stationarity on interior faces and the declared face law."""
        faces = self.facet_faces
        kind, value = boundary.kind[faces], boundary.value[faces]
        result = -outward_sum
        result = jnp.where(kind == 2, result + self.facet_shares * value, result)
        result = jnp.where(
            kind == 3,
            result
            + self.facet_shares * boundary.conductance[faces] * (facet_values - value),
            result,
        )
        return jnp.where(kind == 1, facet_values - value, result)

    def residual(
        self,
        cell_values: ArrayLike,
        facet_values: Array,
        tensor: ArrayLike,
        boundary: HybridDiffusionBoundary,
        *,
        source: ArrayLike = 0.0,
        body_force: ArrayLike | None = None,
    ) -> Array:
        if boundary.geometry_id != self.discretization.geometry_id:
            raise ValueError("Boundary and diffusion geometry must match.")
        local = self.local_fluxes(
            cell_values,
            self._impose_dirichlet(boundary, facet_values),
            tensor,
            body_force=body_force,
        )
        cells = jnp.sum(local, axis=1) - jnp.broadcast_to(
            jnp.asarray(source), (self.cell_count,)
        )
        facets = self._trace_residual(
            boundary, facet_values, self.continuity_residual(local)
        )
        return jnp.concatenate((cells, facets))

    def linear_system(
        self,
        tensor: ArrayLike,
        boundary: HybridDiffusionBoundary,
        *,
        source: ArrayLike = 0.0,
        body_force: ArrayLike | None = None,
    ) -> tuple[LinearSystem, Array]:
        """Build the lifted global hybrid system; shared facets stay global."""
        size = self.cell_count + self.facet_count
        zero = jnp.zeros(size, dtype=self.discretization.cell_volumes.dtype)
        zero = eqx.error_if(
            zero,
            ~jnp.all(self.anchored_components(boundary)),
            "Every diffusion component needs Dirichlet or positive Robin anchoring.",
        )

        def residual(value: Array) -> Array:
            return self.residual(
                value[: self.cell_count],
                value[self.cell_count :],
                tensor,
                boundary,
                source=source,
                body_force=body_force,
            )

        offset = residual(zero)
        space = ArraySpace((size,), dtype=zero.dtype)
        operator = FunctionLinearOperator(
            lambda value: residual(value) - offset,
            source=space,
            target=space,
        )
        return LinearSystem(operator), -offset

    def solve(
        self,
        tensor: ArrayLike,
        boundary: HybridDiffusionBoundary,
        *,
        source: ArrayLike = 0.0,
        body_force: ArrayLike | None = None,
        policy: LinearSolvePolicy | None = None,
    ) -> LinearSolveResult:
        """Solve for cell potentials followed by trace-facet potentials."""
        system, rhs = self.linear_system(
            tensor, boundary, source=source, body_force=body_force
        )
        return solve(
            system,
            rhs,
            policy=(
                LinearSolvePolicy(GMRES(restart=40, stagnation_iterations=40))
                if policy is None
                else policy
            ),
        )


__all__ = ["HybridMimeticDiffusion"]
