#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...ein import contract
from ...linalg import ArraySpace, FunctionLinearOperator, OperatorProperties
from .._cell_complex import TetrahedralConnectivity
from .._cell_mesh import CellMesh


_LOCAL_EDGES = np.asarray(
    ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)), dtype=np.int32
)
_REFERENCE_GRADIENTS = np.asarray(
    ((-1.0, -1.0, -1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
)
_QUADRATURE_BARYCENTRIC = np.asarray(
    (
        (0.5854101966249685, 0.1381966011250105, 0.1381966011250105, 0.1381966011250105),
        (0.1381966011250105, 0.5854101966249685, 0.1381966011250105, 0.1381966011250105),
        (0.1381966011250105, 0.1381966011250105, 0.5854101966249685, 0.1381966011250105),
        (0.1381966011250105, 0.1381966011250105, 0.1381966011250105, 0.5854101966249685),
    )
)


class TetrahedralNedelecSpace(StrictModule, NonTrainableState):
    """Lowest-order first-family H(curl) space on affine tetrahedra.

    Global degrees of freedom are oriented edge line integrals. Covariant Piola
    basis values, physical curls, edge/cell maps, and orientation signs are
    prepared once. The exact topological grad-curl and curl-div sequences use
    the canonical mesh incidence and do not depend on metric material data.
    """

    mesh: CellMesh
    cells: Array
    edges: Array
    cell_edges: Array
    cell_edge_signs: Array
    jacobians: Array
    inverse_transpose_jacobians: Array
    determinants: Array
    volumes: Array
    basis_values: Array
    basis_curls: Array
    edge_space: ArraySpace
    vertex_space: ArraySpace
    face_space: ArraySpace
    cell_space: ArraySpace
    space_id: str = eqx.field(static=True)

    def __init__(self, mesh: CellMesh, /, *, geometry_tolerance: float = 1e-12):
        if not isinstance(mesh, CellMesh):
            raise TypeError("Tetrahedral H(curl) space requires CellMesh.")
        if mesh.topological_dimension != 3 or mesh.ambient_dimension != 3:
            raise ValueError(
                "Tetrahedral H(curl) requires full three-dimensional geometry."
            )
        if any(block.cell_kind != "tetrahedron" for block in mesh.blocks):
            raise ValueError("Tetrahedral H(curl) accepts tetrahedral blocks only.")
        connectivity = mesh.connectivity
        if not isinstance(connectivity, TetrahedralConnectivity):
            raise TypeError("Tetrahedral H(curl) requires tetrahedral connectivity.")
        tolerance = float(geometry_tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("H(curl) geometry tolerance must be positive and finite.")
        cells = np.concatenate(
            [np.asarray(block.vertices, dtype=np.int32) for block in mesh.blocks]
        )
        coordinates = np.asarray(mesh.coordinates, dtype=float)
        edges = np.asarray(connectivity.edges, dtype=np.int32)
        edge_lookup = {tuple(edge): index for index, edge in enumerate(edges)}
        cell_edges = np.empty((cells.shape[0], 6), dtype=np.int32)
        signs = np.empty_like(cell_edges, dtype=float)
        for cell, vertices in enumerate(cells):
            for local, (left, right) in enumerate(_LOCAL_EDGES):
                a, b = int(vertices[left]), int(vertices[right])
                key = (min(a, b), max(a, b))
                cell_edges[cell, local] = edge_lookup[key]
                signs[cell, local] = 1.0 if (a, b) == key else -1.0
        jacobians = np.stack(
            (
                coordinates[cells[:, 1]] - coordinates[cells[:, 0]],
                coordinates[cells[:, 2]] - coordinates[cells[:, 0]],
                coordinates[cells[:, 3]] - coordinates[cells[:, 0]],
            ),
            axis=-1,
        )
        determinants = np.linalg.det(jacobians)
        scale = np.max(np.abs(jacobians), axis=(1, 2))
        if np.any(~np.isfinite(jacobians)) or np.any(
            np.abs(determinants) <= tolerance * scale**3
        ):
            raise ValueError(
                "H(curl) tetrahedra must have finite nonsingular affine Jacobians."
            )
        inverse_transpose = np.linalg.inv(jacobians).transpose((0, 2, 1))
        reference_basis = np.empty((4, 6, 3), dtype=float)
        reference_curl = np.empty((6, 3), dtype=float)
        for edge, (left, right) in enumerate(_LOCAL_EDGES):
            reference_basis[:, edge] = (
                _QUADRATURE_BARYCENTRIC[:, left, None] * _REFERENCE_GRADIENTS[right]
                - _QUADRATURE_BARYCENTRIC[:, right, None] * _REFERENCE_GRADIENTS[left]
            )
            reference_curl[edge] = 2.0 * np.cross(
                _REFERENCE_GRADIENTS[left], _REFERENCE_GRADIENTS[right]
            )
        basis = np.einsum("cij,qej->cqei", inverse_transpose, reference_basis)
        curl = (
            np.einsum("cij,ej->cei", jacobians, reference_curl)
            / determinants[:, None, None]
        )
        basis *= signs[:, None, :, None]
        curl *= signs[:, :, None]
        dtype = np.dtype(coordinates.dtype)
        self.mesh = mesh
        self.cells = jnp.asarray(cells)
        self.edges = jnp.asarray(edges)
        self.cell_edges = jnp.asarray(cell_edges)
        self.cell_edge_signs = jnp.asarray(signs)
        self.jacobians = jnp.asarray(jacobians)
        self.inverse_transpose_jacobians = jnp.asarray(inverse_transpose)
        self.determinants = jnp.asarray(determinants)
        self.volumes = jnp.asarray(np.abs(determinants) / 6.0)
        self.basis_values = jnp.asarray(basis)
        self.basis_curls = jnp.asarray(curl)
        self.edge_space = ArraySpace((edges.shape[0],), dtype=dtype)
        self.vertex_space = ArraySpace((coordinates.shape[0],), dtype=dtype)
        self.face_space = ArraySpace((connectivity.faces.shape[0],), dtype=dtype)
        self.cell_space = ArraySpace((cells.shape[0],), dtype=dtype)
        self.space_id = canonical_fingerprint(
            {
                "kind": "tetrahedral-nedelec-first-family-order-zero",
                "mesh": mesh.mesh_id,
                "cell_edges": cell_edges,
                "cell_edge_signs": signs,
            }
        )

    @property
    def edge_count(self) -> int:
        return self.edge_space.size

    @property
    def cell_count(self) -> int:
        return self.cell_space.size

    def gradient(self, vertex_values: ArrayLike, /) -> Array:
        values = jnp.asarray(vertex_values)
        if values.shape != (self.vertex_space.size,):
            raise ValueError("H(curl) gradient input must have one value per vertex.")
        return values[self.edges[:, 1]] - values[self.edges[:, 0]]

    def discrete_curl(self, edge_integrals: ArrayLike, /) -> Array:
        values = jnp.asarray(edge_integrals)
        if values.shape != (self.edge_count,):
            raise ValueError("Discrete curl input must have one integral per edge.")
        connectivity = self.mesh.connectivity
        if not isinstance(connectivity, TetrahedralConnectivity):
            raise TypeError("Tetrahedral connectivity was lost.")
        return jnp.sum(
            values[connectivity.face_edges] * connectivity.face_edge_signs, axis=1
        )

    def discrete_divergence(self, face_fluxes: ArrayLike, /) -> Array:
        values = jnp.asarray(face_fluxes)
        connectivity = self.mesh.connectivity
        if not isinstance(connectivity, TetrahedralConnectivity):
            raise TypeError("Tetrahedral connectivity was lost.")
        if values.shape != (connectivity.faces.shape[0],):
            raise ValueError("Discrete divergence input must have one flux per face.")
        return jnp.sum(
            values[connectivity.cell_faces] * connectivity.cell_face_signs, axis=1
        )

    def evaluate(self, edge_integrals: ArrayLike, /) -> Array:
        values = jnp.asarray(edge_integrals)
        if values.shape != (self.edge_count,):
            raise ValueError("H(curl) coefficients must have one value per edge.")
        local = values[self.cell_edges]
        return contract("cqei,ce->cqi", self.basis_values, local)

    def curl(self, edge_integrals: ArrayLike, /) -> Array:
        values = jnp.asarray(edge_integrals)
        if values.shape != (self.edge_count,):
            raise ValueError("H(curl) coefficients must have one value per edge.")
        return contract("cei,ce->ci", self.basis_curls, values[self.cell_edges])

    def _tensor(self, value: ArrayLike, name: str) -> Array:
        tensor = jnp.asarray(value)
        if tensor.shape == ():
            tensor = jnp.broadcast_to(tensor * jnp.eye(3), (self.cell_count, 3, 3))
        elif tensor.shape == (self.cell_count,):
            tensor = tensor[:, None, None] * jnp.eye(3)
        elif tensor.shape == (3, 3):
            tensor = jnp.broadcast_to(tensor, (self.cell_count, 3, 3))
        elif tensor.shape != (self.cell_count, 3, 3):
            raise ValueError(f"{name} must be scalar, cell scalar, 3x3, or cell 3x3.")
        tensor = eqx.error_if(
            tensor,
            jnp.any(~jnp.isfinite(tensor))
            | jnp.any(jnp.abs(tensor - jnp.swapaxes(tensor.conj(), -1, -2)) > 1e-10),
            f"{name} must be finite Hermitian tensor data.",
        )
        return tensor

    def mass_action(self, edge_integrals: ArrayLike, tensor: ArrayLike = 1.0, /) -> Array:
        values = jnp.asarray(edge_integrals)
        if values.shape != (self.edge_count,):
            raise ValueError("H(curl) mass action input has wrong shape.")
        material = self._tensor(tensor, "H(curl) mass tensor")
        local_values = self.evaluate(values)
        weighted = contract("cij,cqj->cqi", material, local_values)
        local = (
            self.volumes[:, None]
            * 0.25
            * contract("cqei,cqi->ce", self.basis_values.conj(), weighted)
        )
        return jnp.zeros_like(values).at[self.cell_edges].add(local)

    def curl_curl_action(
        self, edge_integrals: ArrayLike, inverse_permeability: ArrayLike = 1.0, /
    ) -> Array:
        values = jnp.asarray(edge_integrals)
        if values.shape != (self.edge_count,):
            raise ValueError("H(curl) curl-curl input has wrong shape.")
        material = self._tensor(inverse_permeability, "inverse permeability")
        curl_value = self.curl(values)
        weighted = contract("cij,cj->ci", material, curl_value)
        local = self.volumes[:, None] * contract(
            "cei,ci->ce", self.basis_curls.conj(), weighted
        )
        return jnp.zeros_like(values).at[self.cell_edges].add(local)

    def operator(
        self,
        /,
        *,
        mass_tensor: ArrayLike = 0.0,
        curl_tensor: ArrayLike = 1.0,
    ) -> FunctionLinearOperator:
        def action(values):
            return self.curl_curl_action(values, curl_tensor) + self.mass_action(
                values, mass_tensor
            )

        return FunctionLinearOperator(
            action,
            source=self.edge_space,
            target=self.edge_space,
            properties=OperatorProperties(
                self_adjoint=True, evidence={"self_adjoint": "construction"}
            ),
            operator_id=canonical_fingerprint(
                {
                    "kind": "tetrahedral-hcurl-operator",
                    "space": self.space_id,
                    "mass_tensor": np.asarray(mass_tensor),
                    "curl_tensor": np.asarray(curl_tensor),
                }
            ),
        )


__all__ = ["TetrahedralNedelecSpace"]
