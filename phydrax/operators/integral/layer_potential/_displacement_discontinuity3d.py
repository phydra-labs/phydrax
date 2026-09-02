#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import pi

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....linalg import ArraySpace, DenseLinearOperator, OperatorProperties


class DisplacementDiscontinuitySpace3D(StrictModule, NonTrainableState):
    vertices: Array
    faces: Array
    crack_front_edges: Array
    vector_space: ArraySpace
    vertex_count: int = eqx.field(static=True)
    face_count: int = eqx.field(static=True)
    space_id: str = eqx.field(static=True)


class DisplacementDiscontinuityEvidence3D(StrictModule, NonTrainableState):
    minimum_face_area: float = eqx.field(static=True)
    maximum_symmetry_defect: float = eqx.field(static=True)
    rigid_jump_defect: float = eqx.field(static=True)
    resident_bytes: int = eqx.field(static=True)
    conforming_p1: bool = eqx.field(static=True)
    dp0_hypersingular_supported: bool = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class PreparedDisplacementDiscontinuity3D(StrictModule, NonTrainableState):
    space: DisplacementDiscontinuitySpace3D
    traction_operator: DenseLinearOperator
    evidence: DisplacementDiscontinuityEvidence3D
    shear_modulus: float = eqx.field(static=True)
    poisson_ratio: float = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def traction(self, displacement_jump: ArrayLike, /) -> Array:
        value = jnp.asarray(displacement_jump)
        if value.shape == (self.space.vertex_count, 3):
            value = value.reshape((-1,))
        return self.traction_operator.mv(value).reshape((self.space.vertex_count, 3))


def _space(vertices, faces):
    points = np.asarray(vertices, dtype=float)
    triangles = np.asarray(faces, dtype=np.int32)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or triangles.ndim != 2
        or triangles.shape[1] != 3
        or np.any(triangles < 0)
        or np.any(triangles >= points.shape[0])
    ):
        raise ValueError(
            "Displacement-discontinuity geometry requires vertices (n,3) and faces (m,3)."
        )
    edge_uses = {}
    for face in triangles:
        for i in range(3):
            edge = tuple(sorted((int(face[i]), int(face[(i + 1) % 3]))))
            edge_uses[edge] = edge_uses.get(edge, 0) + 1
    front = np.asarray(
        [edge for edge, count in edge_uses.items() if count == 1], dtype=np.int32
    )
    if front.size == 0:
        raise ValueError(
            "Displacement discontinuity requires an open sheet with crack-front edges."
        )
    identifier = canonical_fingerprint(
        {
            "kind": "displacement-discontinuity-space-3d",
            "vertices": array_tree_fingerprint(points),
            "faces": array_tree_fingerprint(triangles),
            "front": array_tree_fingerprint(front),
        }
    )
    return DisplacementDiscontinuitySpace3D(
        jnp.asarray(points),
        jnp.asarray(triangles),
        jnp.asarray(front),
        ArraySpace((3 * points.shape[0],), dtype=jnp.asarray(points).dtype),
        points.shape[0],
        triangles.shape[0],
        identifier,
    )


def prepare_displacement_discontinuity_3d(
    vertices: ArrayLike,
    faces: ArrayLike,
    /,
    *,
    shear_modulus: float,
    poisson_ratio: float,
    maximum_resident_bytes: int = 1_000_000_000,
) -> PreparedDisplacementDiscontinuity3D:
    """Prepare a conforming vector-P1 regularized elastic hypersingular action."""
    space = _space(vertices, faces)
    mu = float(shear_modulus)
    nu = float(poisson_ratio)
    if not np.isfinite(mu) or mu <= 0 or not np.isfinite(nu) or not -1.0 < nu < 0.5:
        raise ValueError("Elastic constants violate the isotropic stability envelope.")
    points = np.asarray(space.vertices)
    faces_ = np.asarray(space.faces)
    tri = points[faces_]
    normals_raw = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    doubled = np.linalg.norm(normals_raw, axis=1)
    if np.any(doubled <= 0):
        raise ValueError("Crack faces must be nondegenerate.")
    areas = 0.5 * doubled
    normals = normals_raw / doubled[:, None]
    centroids = np.mean(tri, axis=1)
    gradients = (
        np.stack(
            (
                np.cross(normals, tri[:, 2] - tri[:, 1]),
                np.cross(normals, tri[:, 0] - tri[:, 2]),
                np.cross(normals, tri[:, 1] - tri[:, 0]),
            ),
            axis=1,
        )
        / doubled[:, None, None]
    )
    scalar = np.empty((space.face_count, space.face_count), dtype=float)
    for target in range(space.face_count):
        for source in range(space.face_count):
            if target == source:
                effective_radius = np.sqrt(areas[target] / pi)
                scalar[target, source] = (
                    areas[target] * areas[source] / (4.0 * pi * effective_radius)
                )
            else:
                radius = np.linalg.norm(centroids[target] - centroids[source])
                scalar[target, source] = (
                    areas[target] * areas[source] / (4.0 * pi * radius)
                )
    vertex_count = space.vertex_count
    scalar_w = np.zeros((vertex_count, vertex_count), dtype=float)
    for target in range(space.face_count):
        for source in range(space.face_count):
            local = scalar[target, source] * (gradients[target] @ gradients[source].T)
            scalar_w[np.ix_(faces_[target], faces_[source])] += local
    lame = 2.0 * mu * nu / (1.0 - 2.0 * nu)
    matrix = np.kron(scalar_w, mu * np.eye(3))
    # The volumetric tangential-divergence contribution is positive and shares
    # the conforming P1 scalar regularization on the declared isotropic envelope.
    matrix += np.kron(scalar_w, (lame + mu) / 3.0 * np.ones((3, 3)))
    matrix = 0.5 * (matrix + matrix.T)
    resident = matrix.nbytes
    if resident > int(maximum_resident_bytes):
        raise ValueError("Displacement-discontinuity resident-byte capacity exceeded.")
    rigid = np.tile(np.eye(3), (vertex_count, 1))
    rigid_defect = float(np.linalg.norm(matrix @ rigid, ord=np.inf))
    symmetry = float(np.linalg.norm(matrix - matrix.T, ord=np.inf))
    evidence_id = canonical_fingerprint(
        {
            "kind": "displacement-discontinuity-evidence-3d",
            "space": space.space_id,
            "mu": mu,
            "nu": nu,
            "matrix": array_tree_fingerprint(matrix),
            "rigid_defect": rigid_defect,
        }
    )
    evidence = DisplacementDiscontinuityEvidence3D(
        float(np.min(areas)), symmetry, rigid_defect, resident, True, False, evidence_id
    )
    operator = DenseLinearOperator(
        jnp.asarray(matrix),
        properties=OperatorProperties(
            self_adjoint=True,
            positive_semidefinite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_semidefinite": "construction",
            },
        ),
        operator_id=f"{evidence_id}:traction",
    )
    return PreparedDisplacementDiscontinuity3D(
        space,
        operator,
        evidence,
        mu,
        nu,
        canonical_fingerprint(
            {"kind": "prepared-displacement-discontinuity-3d", "evidence": evidence_id}
        ),
    )


__all__ = [
    "DisplacementDiscontinuityEvidence3D",
    "DisplacementDiscontinuitySpace3D",
    "PreparedDisplacementDiscontinuity3D",
    "prepare_displacement_discontinuity_3d",
]
