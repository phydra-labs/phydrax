#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Procedural reference geometry and spaces of the pinned 0698e3d executable.

This is the idealized muscle and two aponeurosis sheets, not an anatomical mesh.
Reference axes are fibre, aponeurosis length, width, respectively.
"""

from __future__ import annotations

from itertools import product
from math import asin, cos, isfinite, pi, sin

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....discretization import CellBlock, CellMesh
from ....discretization.fem import (
    FiniteElementDiscretization,
    FiniteElementFieldSpec,
    FiniteElementPlan,
    FiniteElementSpec,
    lagrange_element,
)


def _monomials(points):
    """Total degree one on [0,1]^3: exactly four cell-local coefficients."""
    values = jnp.concatenate((jnp.ones_like(points[:, :1]), points), axis=1)
    gradients = jnp.broadcast_to(
        jnp.concatenate(
            (jnp.zeros((1, 3), dtype=points.dtype), jnp.eye(3, dtype=points.dtype))
        ),
        (points.shape[0], 4, 3),
    )
    return values, gradients


def _dgpm1():
    return FiniteElementSpec(
        "DiscontinuousTotalDegreeMonomial",
        "hexahedron",
        1,
        np.zeros((4, 3)),
        (((),) * 8, ((),) * 12, ((),) * 6, ((0, 1, 2, 3),)),
        conformity="L2",
        representation="modal_coefficient",
        tabulator=_monomials,
        tabulator_id="total-degree-one:1-x-y-z:unit-hexahedron",
    )


class Almonacid2024Geometry(StrictModule, NonTrainableState):
    muscle_length_m: float = eqx.field(static=True)
    aponeurosis_length_m: float = eqx.field(static=True)
    aponeurosis_height_m: float = eqx.field(static=True)
    muscle_width_m: float = eqx.field(static=True)
    pennation_angle_rad: float = eqx.field(static=True)
    refinement: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        muscle_length_m=0.27,
        aponeurosis_length_m=0.208,
        aponeurosis_height_m=0.003,
        muscle_width_m=0.055,
        pennation_angle_rad=15.3 * pi / 180,
        refinement=2,
    ):
        lengths = tuple(
            float(x)
            for x in (
                muscle_length_m,
                aponeurosis_length_m,
                aponeurosis_height_m,
                muscle_width_m,
            )
        )
        angle = float(pennation_angle_rad)
        if any(not isfinite(x) or x <= 0 for x in lengths):
            raise ValueError("Geometry lengths must be finite and positive.")
        if not isfinite(angle) or not 0 < angle < pi / 2:
            raise ValueError("Procedural pennation must be strictly between 0 and pi/2.")
        if not 0 < lengths[0] * sin(angle) / lengths[1] < 1 or lengths[0] <= lengths[1]:
            raise ValueError(
                "Lengths and pennation must define the source muscle triangle."
            )
        if (
            isinstance(refinement, bool)
            or int(refinement) != refinement
            or refinement < 0
        ):
            raise ValueError("refinement must be a nonnegative integer.")
        (
            self.muscle_length_m,
            self.aponeurosis_length_m,
            self.aponeurosis_height_m,
            self.muscle_width_m,
        ) = lengths
        self.pennation_angle_rad = angle
        self.refinement = int(refinement)

    def edges(self):
        angle = self.pennation_angle_rad
        gamma = pi - asin(sin(angle) * self.muscle_length_m / self.aponeurosis_length_m)
        length = self.aponeurosis_length_m * sin(angle + gamma) / sin(angle)
        beta = pi - gamma - angle
        return (
            np.array((length * cos(angle), 0.0, length * sin(angle))),
            np.array(
                (
                    self.aponeurosis_length_m * abs(cos(beta)),
                    0.0,
                    -self.aponeurosis_length_m * sin(beta),
                )
            ),
            np.array((0.0, self.muscle_width_m, 0.0)),
            np.array((0.0, 0.0, self.aponeurosis_height_m)),
        )


class Almonacid2024TraceGeometry(StrictModule, NonTrainableState):
    cells: Array
    basis: Array
    gradients: Array
    scalar_basis: Array
    points_m: Array
    normals: Array
    weights_m2: Array
    nearest_volume_qp: Array
    boundary_ids: Array
    neighbour_cells: Array
    neighbour_basis: Array
    neighbour_gradients: Array
    neighbour_scalar_basis: Array


class PreparedAlmonacid2024Geometry(StrictModule, NonTrainableState):
    discretization: FiniteElementDiscretization
    displacement_dofs: Array
    scalar_dofs: Array
    basis: Array
    gradients: Array
    scalar_basis: Array
    points_m: Array
    weights_m3: Array
    tissue_ids: Array
    reference_directions: Array
    fixed_dofs: Array
    pulling_dofs: Array
    free_dofs: Array
    traces: Almonacid2024TraceGeometry
    cell_count: int = eqx.field(static=True)
    displacement_dof_count: int = eqx.field(static=True)
    free_dof_count: int = eqx.field(static=True)


def prepare_geometry(spec: Almonacid2024Geometry, pulling_face_id: int):
    fibre, apo, width, height = spec.edges()
    n = 1 << spec.refinement
    regions = ((np.zeros(3), fibre, n, 1), (-height, height, 1, 2), (fibre, height, 1, 2))
    corners = np.array(
        (
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (1, 1, 1),
            (0, 1, 1),
        )
    )
    vertices, cells, tissues, jacobians, faces = [], [], [], [], {}
    vertex_ids = {}
    for region, (origin, edge0, count0, tissue) in enumerate(regions):
        counts = np.array((count0, 2 * n, n))
        edges = np.stack((edge0, apo, width), axis=1)
        for cell_index in product(*(range(int(count)) for count in counts)):
            index = np.asarray(cell_index)
            points = origin + ((corners + index) / counts) @ edges.T
            row = []
            for point in points:
                key = tuple(np.rint(point / 1e-10).astype(np.int64))
                if key not in vertex_ids:
                    vertex_ids[key] = len(vertices)
                    vertices.append(point)
                row.append(vertex_ids[key])
            cell = len(cells)
            cells.append(row)
            tissues.append(tissue)
            jacobians.append(edges / counts[None, :])
            for axis in range(3):
                for side in range(2):
                    key = tuple(
                        sorted(row[k] for k in range(8) if corners[k, axis] == side)
                    )
                    boundary = -1
                    if index[axis] == (0 if side == 0 else counts[axis] - 1):
                        if axis == 2:
                            boundary = 2 + side
                        elif axis == 1:
                            boundary = (
                                (0 if region == 1 else 6)
                                if side == 0
                                else (1 if region == 2 else 7)
                            )
                        elif region == 1 and side == 0:
                            boundary = 4
                        elif region == 2 and side == 1:
                            boundary = 5
                    faces.setdefault(key, []).append((cell, axis, side, boundary))
    mesh = CellMesh(
        np.asarray(vertices), (CellBlock("muscle-aponeurosis", "hexahedron", cells),)
    )
    q2, dgpm = lagrange_element("hexahedron", 2), _dgpm1()
    fe = FiniteElementPlan(
        mesh,
        (
            FiniteElementFieldSpec("displacement", q2, component_shape=(3,)),
            FiniteElementFieldSpec("pressure", dgpm),
            FiniteElementFieldSpec("dilation", dgpm),
        ),
    ).prepare()
    x, w = np.polynomial.legendre.leggauss(5)
    x, w = (x + 1) / 2, w / 2
    q = np.asarray(tuple(product(x, repeat=3)))
    qw = np.prod(np.asarray(tuple(product(w, repeat=3))), axis=1)
    volume = fe.evaluate_block_geometry("displacement", 0, mesh.coordinates, q, qw)
    scalar_basis = dgpm.tabulate(q)[0]
    dofs = np.asarray(fe.dof_maps[0].cell_dofs[0])
    jacobians = np.asarray(jacobians)
    inverse = np.linalg.inv(jacobians)
    (
        face_basis,
        face_gradient,
        face_scalar,
        face_points,
        face_weights,
        normals,
        nearest,
    ) = [], [], [], [], [], [], []
    (
        face_cells,
        boundary_ids,
        neighbours,
        neighbour_basis,
        neighbour_gradient,
        neighbour_scalar,
    ) = [], [], [], [], [], []
    fixed, pulling = set(), set()
    q2_nodes = np.asarray(q2.reference_nodes)
    for sides in faces.values():
        owner = sides[0]
        cell, axis, side, boundary = owner
        neighbour = sides[1] if len(sides) == 2 else None
        if neighbour is not None and tissues[cell] == tissues[neighbour[0]]:
            continue
        if neighbour is None and boundary < 0:
            raise ValueError(
                "Unmerged source interface; refusing a disconnected geometry."
            )
        tangent_axes = [k for k in range(3) if k != axis]
        ref = np.zeros((25, 3))
        ref[:, axis] = side
        ref[:, tangent_axes] = np.asarray(tuple(product(x, repeat=2)))
        values, grad = q2.tabulate(ref)
        shape = np.asarray(values)
        grad = np.asarray(grad) @ inverse[cell]
        points = (
            np.asarray(vertices)[np.asarray(cells[cell])[0]] + ref @ jacobians[cell].T
        )
        normal = inverse[cell][axis] * (2 * side - 1)
        normal /= np.linalg.norm(normal)
        area = np.linalg.norm(
            np.cross(
                jacobians[cell][:, tangent_axes[0]], jacobians[cell][:, tangent_axes[1]]
            )
        )
        face_basis.append(shape)
        face_gradient.append(grad)
        face_scalar.append(np.asarray(dgpm.tabulate(ref)[0]))
        face_points.append(points)
        normals.append(np.broadcast_to(normal, (25, 3)))
        face_weights.append(
            area * np.prod(np.asarray(tuple(product(w, repeat=2))), axis=1)
        )
        distance = np.sum(
            (points[:, None, :] - np.asarray(volume.physical_points)[cell][None, :, :])
            ** 2,
            axis=-1,
        )
        nearest.append(np.argmin(distance, axis=1))
        face_cells.append(cell)
        boundary_ids.append(boundary)
        neighbours.append(-1 if neighbour is None else neighbour[0])
        if neighbour is None:
            neighbour_basis.append(shape)
            neighbour_gradient.append(grad)
            neighbour_scalar.append(face_scalar[-1])
        else:
            nc = neighbour[0]
            nref = (points - np.asarray(vertices)[np.asarray(cells[nc])[0]]) @ inverse[
                nc
            ].T
            nv, ng = q2.tabulate(nref)
            neighbour_basis.append(np.asarray(nv))
            neighbour_gradient.append(np.asarray(ng) @ inverse[nc])
            neighbour_scalar.append(np.asarray(dgpm.tabulate(nref)[0]))
        boundary_nodes = dofs[cell, np.isclose(q2_nodes[:, axis], side)]
        if boundary == 0:
            fixed.update(int(d) for d in boundary_nodes)
        if boundary == pulling_face_id:
            pulling.update(int(d) for d in boundary_nodes)
    if not fixed or not pulling or fixed & pulling:
        raise ValueError("Fixed and pulling boundary DOFs must be nonempty and disjoint.")
    free = sorted(set(range(fe.dof_maps[0].global_dof_count)) - fixed - pulling)
    beta_apo = (
        asin(
            spec.muscle_length_m
            * sin(spec.pennation_angle_rad)
            / spec.aponeurosis_length_m
        )
        - spec.pennation_angle_rad
    )
    directions = np.asarray(
        [
            (cos(spec.pennation_angle_rad), 0.0, sin(spec.pennation_angle_rad))
            if t == 1
            else (cos(beta_apo), 0.0, -sin(beta_apo))
            for t in tissues
        ]
    )
    trace = Almonacid2024TraceGeometry(
        *map(
            jnp.asarray,
            (
                face_cells,
                face_basis,
                face_gradient,
                face_scalar,
                face_points,
                normals,
                face_weights,
                nearest,
                boundary_ids,
                neighbours,
                neighbour_basis,
                neighbour_gradient,
                neighbour_scalar,
            ),
        )
    )
    return PreparedAlmonacid2024Geometry(
        fe,
        jnp.asarray(dofs),
        fe.dof_maps[1].cell_dofs[0],
        volume.basis_values,
        volume.physical_gradients,
        scalar_basis,
        volume.physical_points,
        volume.physical_weights,
        jnp.asarray(tissues, dtype=jnp.int32),
        jnp.asarray(directions),
        jnp.asarray(sorted(fixed), dtype=jnp.int32),
        jnp.asarray(sorted(pulling), dtype=jnp.int32),
        jnp.asarray(free, dtype=jnp.int32),
        trace,
        len(cells),
        fe.dof_maps[0].global_dof_count,
        len(free),
    )
