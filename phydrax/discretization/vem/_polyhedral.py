#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import FunctionLinearOperator, OperatorProperties
from .._cell_complex import PolyhedralConnectivity
from .._cell_mesh import CellMesh
from ._operator import FactorizedVirtualElementOperator
from ._precision import VirtualElementResourceBudget


class PolyhedralVEMEvidence3D(StrictModule, NonTrainableState):
    """Admissibility and polynomial-consistency evidence for degree-one H1 VEM."""

    cell_volumes: Array
    projector_rank_margins: Array
    polynomial_reproduction_defects: Array
    minimum_volume: float = eqx.field(static=True)
    minimum_rank_margin: float = eqx.field(static=True)
    maximum_reproduction_defect: float = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class PreparedPolyhedralH1VirtualElement3D(StrictModule, NonTrainableState):
    """Matrix-free conforming degree-one H1 VEM on root polyhedral topology."""

    mesh: CellMesh
    operator: FactorizedVirtualElementOperator
    evidence: PolyhedralVEMEvidence3D
    degree: int = eqx.field(static=True)
    dof_count: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def mv(self, values: ArrayLike, /) -> Array:
        return self.operator.mv(values)

    def transpose_mv(self, values: ArrayLike, /) -> Array:
        return self.operator.transpose_mv(values)

    def as_linear_operator(self, /) -> FunctionLinearOperator:
        return self.operator.as_linear_operator()


def _cell_matrix(
    coordinates: np.ndarray,
    connectivity: PolyhedralConnectivity,
    cell_index: int,
    /,
) -> tuple[np.ndarray, float, float, float]:
    cell_faces = np.asarray(connectivity.cell_faces)
    cell_face_valid = np.asarray(connectivity.cell_face_valid)
    cell_face_signs = np.asarray(connectivity.cell_face_signs)
    face_vertices = np.asarray(connectivity.face_vertices)
    face_vertex_valid = np.asarray(connectivity.face_vertex_valid)
    cell_vertices = np.asarray(connectivity.cell_vertices)
    cell_vertex_valid = np.asarray(connectivity.cell_vertex_valid)
    vertices = cell_vertices[cell_index, cell_vertex_valid[cell_index]]
    local = {int(vertex): index for index, vertex in enumerate(vertices)}
    points = coordinates[vertices]
    centroid = np.mean(points, axis=0)
    characteristic_length = float(
        np.max(np.linalg.norm(points - centroid[None, :], axis=1))
    )
    if not np.isfinite(characteristic_length) or characteristic_length <= 0.0:
        raise ValueError("Polyhedral VEM cells require positive diameter.")

    gradients = np.zeros((3, vertices.size), dtype=float)
    volume = 0.0
    for face_index, sign in zip(
        cell_faces[cell_index, cell_face_valid[cell_index]],
        cell_face_signs[cell_index, cell_face_valid[cell_index]],
        strict=True,
    ):
        face = face_vertices[face_index, face_vertex_valid[face_index]]
        oriented = face if sign > 0.0 else face[::-1]
        face_points = coordinates[oriented]
        for offset in range(1, oriented.size - 1):
            triangle_vertices = (
                int(oriented[0]),
                int(oriented[offset]),
                int(oriented[offset + 1]),
            )
            first, second, third = (
                face_points[0],
                face_points[offset],
                face_points[offset + 1],
            )
            area_vector = 0.5 * np.cross(second - first, third - first)
            signed_volume = float(np.dot(first - centroid, area_vector) / 3.0)
            if not np.isfinite(signed_volume) or signed_volume <= 0.0:
                raise ValueError(
                    "Polyhedral VEM requires outward faces visible from the cell centroid."
                )
            volume += signed_volume
            for vertex in triangle_vertices:
                gradients[:, local[vertex]] += area_vector / 3.0
    if not np.isfinite(volume) or volume <= 0.0:
        raise ValueError("Polyhedral VEM requires positive finite cell volume.")
    gradients /= volume
    consistency = volume * gradients.T @ gradients

    monomials = np.concatenate(
        (
            np.ones((vertices.size, 1), dtype=float),
            (points - centroid[None, :]) / characteristic_length,
        ),
        axis=1,
    )
    singular_values = np.linalg.svd(monomials, compute_uv=False)
    rank_margin = float(singular_values[-1])
    if rank_margin <= np.finfo(float).eps * singular_values[0] * vertices.size:
        raise ValueError("Polyhedral VEM affine projector is rank deficient.")
    gram_inverse = np.linalg.inv(monomials.T @ monomials)
    projector = monomials @ gram_inverse @ monomials.T
    kernel = np.eye(vertices.size) - projector
    scale = float(np.trace(consistency) / max(vertices.size, 1))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = volume ** (1.0 / 3.0)
    stabilization = scale * (kernel.T @ kernel)
    reproduction = float(np.linalg.norm(kernel @ monomials, ord=np.inf))
    return consistency + stabilization, volume, rank_margin, reproduction


def prepare_polyhedral_h1_virtual_element_3d(
    mesh: CellMesh,
    /,
    *,
    degree: int = 1,
    resource_budget: VirtualElementResourceBudget | None = None,
    accumulation: str = "fast",
) -> PreparedPolyhedralH1VirtualElement3D:
    """Prepare the first root-topology 3-D H1 VEM consumer.

    The supported tuple is deliberately exact: affine geometry, star-visible
    oriented polyhedra, and degree one. Higher-degree 2-D VEM remains available
    through ``VirtualElementPlan``; unsupported 3-D degree requests fail rather
    than substituting a low-order cell.
    """

    if not isinstance(mesh, CellMesh):
        raise TypeError("mesh must be CellMesh.")
    if mesh.topological_dimension != 3 or mesh.ambient_dimension != 3:
        raise ValueError("Polyhedral H1 VEM requires a 3-D CellMesh in R3.")
    if not isinstance(mesh.connectivity, PolyhedralConnectivity):
        raise TypeError("Polyhedral H1 VEM requires root PolyhedralConnectivity.")
    if int(degree) != 1:
        raise ValueError("The qualified polyhedral H1 VEM tuple has degree one.")
    budget = (
        VirtualElementResourceBudget() if resource_budget is None else resource_budget
    )
    if not isinstance(budget, VirtualElementResourceBudget):
        raise TypeError("resource_budget must be VirtualElementResourceBudget.")
    connectivity = mesh.connectivity
    if connectivity.cell_count > budget.maximum_cells:
        raise ValueError("Polyhedral VEM cell capacity exceeded.")

    coordinates = np.asarray(mesh.coordinates, dtype=float)
    cell_vertices = np.asarray(connectivity.cell_vertices)
    cell_vertex_valid = np.asarray(connectivity.cell_vertex_valid)
    buckets: dict[int, list[int]] = {}
    matrices: dict[int, list[np.ndarray]] = {}
    volumes = np.empty((connectivity.cell_count,), dtype=float)
    rank_margins = np.empty_like(volumes)
    defects = np.empty_like(volumes)
    estimated_bytes = 0
    for cell_index in range(connectivity.cell_count):
        arity = int(np.sum(cell_vertex_valid[cell_index]))
        if arity > budget.maximum_local_dofs:
            raise ValueError("Polyhedral VEM local-DOF capacity exceeded.")
        matrix, volume, margin, defect = _cell_matrix(
            coordinates, connectivity, cell_index
        )
        buckets.setdefault(arity, []).append(cell_index)
        matrices.setdefault(arity, []).append(matrix)
        volumes[cell_index] = volume
        rank_margins[cell_index] = margin
        defects[cell_index] = defect
        estimated_bytes += int(matrix.nbytes * 3)
    if estimated_bytes > budget.maximum_projector_bytes:
        raise ValueError("Polyhedral VEM projector byte capacity exceeded.")

    coefficient_maps = []
    polynomial_matrices = []
    stabilization_matrices = []
    gathers = []
    for arity in sorted(buckets):
        indices = np.asarray(buckets[arity], dtype=np.int32)
        local = np.stack(matrices[arity])
        coefficient_maps.append(
            np.broadcast_to(np.eye(arity), (indices.size, arity, arity)).copy()
        )
        polynomial_matrices.append(local)
        stabilization_matrices.append(np.zeros_like(local))
        gathers.append(cell_vertices[indices, :arity])
    properties = OperatorProperties(
        self_adjoint=True,
        positive_semidefinite=True,
        evidence={
            "self_adjoint": "construction",
            "positive_semidefinite": "construction",
        },
    )
    operator = FactorizedVirtualElementOperator(
        tuple(coefficient_maps),
        tuple(polynomial_matrices),
        tuple(stabilization_matrices),
        tuple(gathers),
        connectivity.vertex_count,
        accumulation=accumulation,
        properties=properties,
        operator_id=canonical_fingerprint(
            {
                "kind": "polyhedral-h1-vem-operator",
                "mesh": mesh.topology_id,
                "degree": 1,
                "matrices": array_tree_fingerprint(tuple(polynomial_matrices)),
            }
        ),
    )
    evidence_id = canonical_fingerprint(
        {
            "kind": "polyhedral-h1-vem-evidence",
            "mesh": mesh.geometry_id,
            "volumes": array_tree_fingerprint(volumes),
            "rank_margins": array_tree_fingerprint(rank_margins),
            "defects": array_tree_fingerprint(defects),
        }
    )
    evidence = PolyhedralVEMEvidence3D(
        cell_volumes=jnp.asarray(volumes),
        projector_rank_margins=jnp.asarray(rank_margins),
        polynomial_reproduction_defects=jnp.asarray(defects),
        minimum_volume=float(np.min(volumes)),
        minimum_rank_margin=float(np.min(rank_margins)),
        maximum_reproduction_defect=float(np.max(defects)),
        evidence_id=evidence_id,
    )
    return PreparedPolyhedralH1VirtualElement3D(
        mesh=mesh,
        operator=operator,
        evidence=evidence,
        degree=1,
        dof_count=connectivity.vertex_count,
        prepared_id=canonical_fingerprint(
            {
                "kind": "prepared-polyhedral-h1-vem",
                "mesh": mesh.mesh_id,
                "operator": operator.operator_id,
                "evidence": evidence_id,
            }
        ),
    )


__all__ = [
    "PolyhedralVEMEvidence3D",
    "PreparedPolyhedralH1VirtualElement3D",
    "prepare_polyhedral_h1_virtual_element_3d",
]
