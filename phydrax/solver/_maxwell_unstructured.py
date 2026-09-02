#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import (
    CochainDiscretization,
    tetrahedral_cell_complex,
    tetrahedral_connectivity,
)
from ._maxwell import (
    AbstractMaxwellConstitutivePlan,
    CompatibleMaxwellState,
    MaxwellAuxiliaryState,
    MaxwellCochainLayout,
    MaxwellPrimaryState,
)


class TetrahedralMaxwellQuality(StrictModule, NonTrainableState):
    minimum_volume: float = eqx.field(static=True)
    maximum_aspect_ratio: float = eqx.field(static=True)
    electric_minimum_eigenvalue: float = eqx.field(static=True)
    magnetic_minimum_eigenvalue: float = eqx.field(static=True)
    passed: bool = eqx.field(static=True)
    quality_id: str = eqx.field(static=True)


class TetrahedralMaxwellHodge(StrictModule, NonTrainableState):
    cochain: CochainDiscretization
    electric_mass: Array
    magnetic_mass: Array
    quality: TetrahedralMaxwellQuality
    hodge_id: str = eqx.field(static=True)


_QUAD_A = 0.5854101966249685
_QUAD_B = 0.1381966011250105
_QUADRATURE = np.asarray(
    (
        (_QUAD_A, _QUAD_B, _QUAD_B, _QUAD_B),
        (_QUAD_B, _QUAD_A, _QUAD_B, _QUAD_B),
        (_QUAD_B, _QUAD_B, _QUAD_A, _QUAD_B),
        (_QUAD_B, _QUAD_B, _QUAD_B, _QUAD_A),
    )
)
_LOCAL_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
_LOCAL_FACES = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))


def tetrahedral_maxwell_hodge(
    vertices: ArrayLike,
    tetrahedra: ArrayLike,
    /,
    *,
    permittivity: float = 1.0,
    inverse_permeability: float = 1.0,
) -> TetrahedralMaxwellHodge:
    points = np.asarray(vertices, dtype=float)
    cells = np.asarray(tetrahedra, dtype=np.int32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("vertices must have shape (vertices, 3).")
    if cells.ndim != 2 or cells.shape[1] != 4:
        raise ValueError("tetrahedra must have shape (cells, 4).")
    if not np.isfinite(permittivity) or permittivity <= 0.0:
        raise ValueError("permittivity must be finite and positive.")
    if not np.isfinite(inverse_permeability) or inverse_permeability <= 0.0:
        raise ValueError("inverse_permeability must be finite and positive.")
    connectivity = tetrahedral_connectivity(cells, points.shape[0])
    topology = tetrahedral_cell_complex(cells, points.shape[0])
    edges = np.asarray(connectivity.edges)
    faces = np.asarray(connectivity.faces)
    edge_lookup = {tuple(edge): index for index, edge in enumerate(edges)}
    face_lookup = {tuple(sorted(face)): index for index, face in enumerate(faces)}
    electric_mass = np.zeros((edges.shape[0], edges.shape[0]))
    magnetic_mass = np.zeros((faces.shape[0], faces.shape[0]))
    vertex_dual = np.zeros(points.shape[0])
    cell_volume = np.empty(cells.shape[0])
    aspect = np.empty(cells.shape[0])
    for cell_index, tetrahedron in enumerate(cells):
        local_points = points[tetrahedron]
        jacobian = np.stack(
            (
                local_points[1] - local_points[0],
                local_points[2] - local_points[0],
                local_points[3] - local_points[0],
            ),
            axis=1,
        )
        determinant = np.linalg.det(jacobian)
        volume = abs(determinant) / 6.0
        if volume <= np.finfo(float).eps:
            raise ValueError("Tetrahedral Maxwell mesh contains a degenerate cell.")
        cell_volume[cell_index] = volume
        gradient = np.empty((4, 3))
        gradient[1:] = np.linalg.solve(
            jacobian.T,
            np.eye(3, dtype=jacobian.dtype),
        )
        gradient[0] = -np.sum(gradient[1:], axis=0)
        lengths = np.asarray(
            [np.linalg.norm(local_points[j] - local_points[i]) for i, j in _LOCAL_EDGES]
        )
        aspect[cell_index] = np.max(lengths) / np.min(lengths)
        vertex_dual[tetrahedron] += volume / 4.0
        local_edge_ids = []
        local_edge_signs = []
        for i, j in _LOCAL_EDGES:
            pair = (int(tetrahedron[i]), int(tetrahedron[j]))
            canonical = (min(pair), max(pair))
            local_edge_ids.append(edge_lookup[canonical])
            local_edge_signs.append(1.0 if pair == canonical else -1.0)
        local_face_ids = []
        local_face_signs = []
        for i, j, k in _LOCAL_FACES:
            oriented = (int(tetrahedron[i]), int(tetrahedron[j]), int(tetrahedron[k]))
            canonical = tuple(sorted(oriented))
            local_face_ids.append(face_lookup[canonical])
            inversions = sum(
                oriented[a] > oriented[b] for a in range(3) for b in range(a + 1, 3)
            )
            local_face_signs.append(-1.0 if inversions % 2 else 1.0)
        local_edge_mass = np.zeros((6, 6))
        local_face_mass = np.zeros((4, 4))
        for barycentric in _QUADRATURE:
            edge_forms = np.asarray(
                [
                    barycentric[i] * gradient[j] - barycentric[j] * gradient[i]
                    for i, j in _LOCAL_EDGES
                ]
            )
            face_forms = np.asarray(
                [
                    2.0
                    * (
                        barycentric[i] * np.cross(gradient[j], gradient[k])
                        + barycentric[j] * np.cross(gradient[k], gradient[i])
                        + barycentric[k] * np.cross(gradient[i], gradient[j])
                    )
                    for i, j, k in _LOCAL_FACES
                ]
            )
            local_edge_mass += permittivity * volume * (edge_forms @ edge_forms.T) / 4.0
            local_face_mass += (
                inverse_permeability * volume * (face_forms @ face_forms.T) / 4.0
            )
        for local_i, global_i in enumerate(local_edge_ids):
            for local_j, global_j in enumerate(local_edge_ids):
                electric_mass[global_i, global_j] += (
                    local_edge_signs[local_i]
                    * local_edge_signs[local_j]
                    * local_edge_mass[local_i, local_j]
                )
        for local_i, global_i in enumerate(local_face_ids):
            for local_j, global_j in enumerate(local_face_ids):
                magnetic_mass[global_i, global_j] += (
                    local_face_signs[local_i]
                    * local_face_signs[local_j]
                    * local_face_mass[local_i, local_j]
                )
    electric_eigenvalues = np.linalg.eigvalsh(electric_mass)
    magnetic_eigenvalues = np.linalg.eigvalsh(magnetic_mass)
    electric_minimum = float(electric_eigenvalues[0])
    magnetic_minimum = float(magnetic_eigenvalues[0])
    passed = electric_minimum > 0.0 and magnetic_minimum > 0.0
    if not passed:
        raise ValueError("Whitney Maxwell mass matrices are not positive definite.")
    edge_diagonal = np.diag(electric_mass)
    face_diagonal = np.diag(magnetic_mass)
    cell_measure = cell_volume
    hodge = (
        np.maximum(vertex_dual, np.finfo(float).eps),
        edge_diagonal,
        face_diagonal,
        np.maximum(cell_measure, np.finfo(float).eps),
    )
    coordinates = (
        points,
        0.5 * (points[edges[:, 0]] + points[edges[:, 1]]),
        np.mean(points[faces], axis=1),
        np.mean(points[cells], axis=1),
    )
    boundary_masks = (
        np.asarray(connectivity.boundary_vertices),
        np.asarray(connectivity.boundary_edges),
        np.asarray(connectivity.boundary_faces),
        np.zeros(cells.shape[0], dtype=bool),
    )
    cochain = CochainDiscretization(
        topology,
        hodge,
        hodge_matrices=(None, electric_mass, magnetic_mass, None),
        primal_measures=(
            np.ones(points.shape[0]),
            np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1),
            np.asarray(
                [
                    0.5
                    * np.linalg.norm(
                        np.cross(
                            points[face[1]] - points[face[0]],
                            points[face[2]] - points[face[0]],
                        )
                    )
                    for face in faces
                ]
            ),
            cell_volume,
        ),
        boundary_masks=boundary_masks,
        coordinates=coordinates,
    )
    quality_id = canonical_fingerprint(
        {
            "kind": "tetrahedral-maxwell-quality",
            "minimum_volume": float(np.min(cell_volume)),
            "maximum_aspect_ratio": float(np.max(aspect)),
            "electric_minimum": electric_minimum,
            "magnetic_minimum": magnetic_minimum,
        }
    )
    quality = TetrahedralMaxwellQuality(
        minimum_volume=float(np.min(cell_volume)),
        maximum_aspect_ratio=float(np.max(aspect)),
        electric_minimum_eigenvalue=electric_minimum,
        magnetic_minimum_eigenvalue=magnetic_minimum,
        passed=passed,
        quality_id=quality_id,
    )
    return TetrahedralMaxwellHodge(
        cochain=cochain,
        electric_mass=jnp.asarray(electric_mass),
        magnetic_mass=jnp.asarray(magnetic_mass),
        quality=quality,
        hodge_id=canonical_fingerprint(
            {
                "kind": "tetrahedral-maxwell-hodge",
                "cochain": cochain.prepared_id,
                "quality": quality_id,
            }
        ),
    )


class CochainPartition(StrictModule, NonTrainableState):
    owners: tuple[Array, ...]
    partition_count: int = eqx.field(static=True)
    partition_id: str = eqx.field(static=True)

    def __init__(self, owners: Sequence[ArrayLike], partition_count: int, /):
        count = int(partition_count)
        arrays = tuple(jnp.asarray(value, dtype=jnp.int32) for value in owners)
        if count <= 0 or not arrays:
            raise ValueError("Cochain partition inputs are invalid.")
        if any(value.ndim != 1 for value in arrays):
            raise ValueError("Cochain owner arrays must be vectors.")
        if any(bool(jnp.any((value < 0) | (value >= count))) for value in arrays):
            raise ValueError("Cochain owner indices are outside partition_count.")
        self.owners = arrays
        self.partition_count = count
        self.partition_id = canonical_fingerprint(
            {
                "kind": "cochain-partition",
                "owners": [array_tree_fingerprint(value) for value in arrays],
                "partition_count": count,
            }
        )


class CochainHaloExchange(StrictModule, NonTrainableState):
    source_degree: int = eqx.field(static=True)
    source_indices: Array
    target_partitions: Array
    exchange_id: str = eqx.field(static=True)

    def __init__(
        self,
        cochain: CochainDiscretization,
        partition: CochainPartition,
        degree: int,
        /,
    ):
        degree_ = int(degree)
        if len(partition.owners) != len(cochain.cell_counts):
            raise ValueError("Partition must cover every cochain degree.")
        if degree_ < 0 or degree_ >= cochain.max_degree:
            raise ValueError("Halo exchange degree must have a following incidence.")
        incidence = cochain.topology.incidences[degree_]
        valid = np.asarray(incidence.relation.valid, dtype=bool)
        sources = np.asarray(incidence.relation.source_indices)[valid]
        targets = np.asarray(incidence.relation.target_indices)[valid]
        source_owner = np.asarray(partition.owners[degree_])[sources]
        target_owner = np.asarray(partition.owners[degree_ + 1])[targets]
        remote = source_owner != target_owner
        self.source_degree = degree_
        self.source_indices = jnp.asarray(sources[remote], dtype=jnp.int32)
        self.target_partitions = jnp.asarray(target_owner[remote], dtype=jnp.int32)
        self.exchange_id = canonical_fingerprint(
            {
                "kind": "cochain-halo-exchange",
                "cochain": cochain.prepared_id,
                "partition": partition.partition_id,
                "degree": degree_,
                "source": array_tree_fingerprint(sources[remote]),
                "target_partition": array_tree_fingerprint(target_owner[remote]),
            }
        )

    def payload(self, values: ArrayLike, /) -> tuple[Array, Array]:
        value = jnp.asarray(values)
        return self.target_partitions, value[self.source_indices]


class UnstructuredMaxwellPlan(StrictModule):
    """Compatible D/B evolution on an arbitrary three-dimensional cochain complex."""

    cochain: CochainDiscretization
    layout: MaxwellCochainLayout
    constitutive: AbstractMaxwellConstitutivePlan
    spectral_upper_bound: float = eqx.field(static=True)
    courant_factor: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        cochain: CochainDiscretization,
        constitutive: AbstractMaxwellConstitutivePlan,
        spectral_upper_bound: float,
        /,
        *,
        courant_factor: float = 0.9,
    ):
        if not isinstance(cochain, CochainDiscretization) or cochain.max_degree != 3:
            raise TypeError("Unstructured Maxwell requires a 3-D CochainDiscretization.")
        if not isinstance(constitutive, AbstractMaxwellConstitutivePlan):
            raise TypeError("constitutive must be AbstractMaxwellConstitutivePlan.")
        bound = float(spectral_upper_bound)
        factor = float(courant_factor)
        if not np.isfinite(bound) or bound <= 0.0:
            raise ValueError("spectral_upper_bound must be finite and positive.")
        if not np.isfinite(factor) or factor <= 0.0 or factor > 1.0:
            raise ValueError("courant_factor must lie in (0, 1].")
        layout = MaxwellCochainLayout(cochain, "full_3d")
        self.cochain = cochain
        self.layout = layout
        self.constitutive = constitutive
        self.spectral_upper_bound = bound
        self.courant_factor = factor
        self.plan_id = canonical_fingerprint(
            {
                "kind": "unstructured-maxwell-plan",
                "cochain": cochain.prepared_id,
                "layout": layout.layout_id,
                "constitutive": constitutive.plan_id,
                "spectral_upper_bound": bound,
                "courant_factor": factor,
            }
        )

    def prepare(self, /) -> PreparedUnstructuredMaxwell:
        return PreparedUnstructuredMaxwell(self)


class PreparedUnstructuredMaxwell(StrictModule):
    plan: UnstructuredMaxwellPlan
    constitutive: Any
    stable_dt: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: UnstructuredMaxwellPlan, /):
        constitutive = plan.constitutive.prepare(plan.cochain, plan.layout)
        self.plan = plan
        self.constitutive = constitutive
        self.stable_dt = (
            plan.courant_factor * 2.0 / jnp.sqrt(jnp.asarray(plan.spectral_upper_bound))
        )
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-unstructured-maxwell",
                "plan": plan.plan_id,
                "constitutive": constitutive.prepared_id,
            }
        )

    def initialize(self, /) -> CompatibleMaxwellState:
        counts = self.plan.cochain.cell_counts
        return CompatibleMaxwellState(
            MaxwellPrimaryState(
                jnp.zeros((counts[1],)),
                jnp.zeros((counts[2],)),
                jnp.zeros((counts[0],)),
            ),
            MaxwellAuxiliaryState(self.constitutive.initialize_state(), None),
            (),
        )

    def electric_field(self, state: CompatibleMaxwellState, /) -> Array:
        return self.constitutive.electric_field(
            state.primary.electric_displacement,
            state.auxiliary.material,
        )

    def magnetic_field(self, state: CompatibleMaxwellState, /) -> Array:
        return self.constitutive.magnetic_field(
            state.primary.magnetic_flux,
            state.auxiliary.material,
        )

    def step(
        self,
        time: ArrayLike,
        state: CompatibleMaxwellState,
        step_size: ArrayLike,
        /,
        *,
        electric_current: ArrayLike | None = None,
    ) -> CompatibleMaxwellState:
        dt = jnp.asarray(step_size)
        dt = eqx.error_if(
            dt,
            ~jnp.isfinite(dt) | (dt <= 0.0) | (dt > self.stable_dt),
            "Unstructured Maxwell step exceeds its stable bound.",
        )
        half = 0.5 * dt
        electric = self.electric_field(state)
        magnetic_half = (
            state.primary.magnetic_flux
            - half * self.plan.cochain.exterior_derivative(1, electric)
        )
        magnetic = self.constitutive.magnetic_field(
            magnetic_half, state.auxiliary.material
        )
        current = (
            jnp.zeros_like(state.primary.electric_displacement)
            if electric_current is None
            else jnp.asarray(
                electric_current, dtype=state.primary.electric_displacement.dtype
            )
        )
        if current.shape != state.primary.electric_displacement.shape:
            raise ValueError("Unstructured Maxwell current must be a degree-one cochain.")
        displacement = state.primary.electric_displacement + dt * (
            self.plan.cochain.codifferential(2, magnetic) - current
        )
        electric_new = self.constitutive.electric_field(
            displacement, state.auxiliary.material
        )
        magnetic_new = magnetic_half - half * self.plan.cochain.exterior_derivative(
            1, electric_new
        )
        del time
        charge = state.primary.charge - dt * self.plan.cochain.codifferential(1, current)
        return CompatibleMaxwellState(
            MaxwellPrimaryState(displacement, magnetic_new, charge),
            state.auxiliary,
            state.observations,
        )

    def constraints(self, state: CompatibleMaxwellState, /) -> tuple[Array, Array]:
        return (
            self.plan.cochain.codifferential(1, state.primary.electric_displacement)
            - state.primary.charge,
            self.plan.cochain.exterior_derivative(2, state.primary.magnetic_flux),
        )


__all__ = [
    "CochainHaloExchange",
    "CochainPartition",
    "PreparedUnstructuredMaxwell",
    "TetrahedralMaxwellHodge",
    "TetrahedralMaxwellQuality",
    "UnstructuredMaxwellPlan",
    "tetrahedral_maxwell_hodge",
]
