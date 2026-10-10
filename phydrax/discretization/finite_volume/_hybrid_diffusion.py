#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Affine-consistent, coercive hybrid mimetic diffusion on admissible 3-D FV cells."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import fixed_field
from ...ein import contract
from ...linalg import (
    ArraySpace,
    ComposedLinearOperator,
    ConjugateGradient,
    DiagonalLinearOperator,
    FunctionLinearOperator,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSystem,
    OperatorProperties,
    solve,
    SumLinearOperator,
)
from ...sparse import EdgeRelation, SparseLinearMap
from ...typing import checked
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


def _trace_cell_rhs_transpose(
    matrices: Array,
    accumulation: SparseLinearMap,
    boundary_kind: Array,
    local_shape: tuple[int, int],
    cotangent: Array,
    /,
) -> Array:
    """Exact local-energy transpose over the owning prepared sparse routes."""
    face_cotangent = jnp.where(boundary_kind == 1, 0, cotangent)
    local_cotangent = accumulation.transpose_mv(face_cotangent).reshape(local_shape)
    return jnp.sum(contract("cfg,cf->cg", matrices, local_cotangent), axis=1)


class HybridMimeticDiffusion(StrictModule):
    """Fixed polyhedral geometry, shared global face potentials, no TPFA reduction.

    Cells must be strictly star-shaped about their FV centers, have planar faces,
    positive measures, outward closure and exact first geometric moments. The
    local energy is V grad(u).K.grad(u) plus a positive normal-distance weighted
    penalty on deviations from affine face traces. This is consistent for full
    rotated SPD tensors, including off-diagonal tensor entries.
    """

    discretization: UnstructuredFiniteVolumeDiscretization
    source_face_indices: Array
    owner_cells: Array
    neighbor_cells: Array
    face_measures: Array
    face_centers: Array
    cell_faces: Array
    valid: Array
    facet_accumulation: SparseLinearMap = fixed_field()
    outward_areas: Array
    displacements: Array
    gradient_weights: Array
    defect: Array
    stabilization_weights: Array
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
        face_centers = np.asarray(discretization.face_centers)
        areas = np.asarray(discretization.area_vectors)
        measures = np.asarray(discretization.face_measures)
        active = np.asarray(discretization.face_block.active_mask)
        maps = np.asarray(discretization.neighbor_frame_maps)
        neighbor_centers = (
            np.einsum("fij,fj->fi", maps[:, :3, :3], face_centers) + maps[:, :3, 3]
        )
        neighbor_areas = -np.einsum("fij,fj->fi", maps[:, :3, :3], areas)
        count = volumes.size
        lists = [[] for _ in range(count)]
        signs = [[] for _ in range(count)]
        slots = np.zeros(owner.size, dtype=np.int32)
        for face, (left, right) in enumerate(zip(owner, neighbor, strict=True)):
            if not active[face]:
                continue
            slots[face] = len(lists[left])
            lists[left].append(face)
            signs[left].append(1)
            if right >= 0:
                lists[right].append(face)
                signs[right].append(-1)
        width = max(map(len, lists))
        faces = np.zeros((count, width), dtype=np.int32)
        orientation = np.zeros((count, width))
        for cell, (indices, directions) in enumerate(zip(lists, signs, strict=True)):
            faces[cell, : len(indices)] = indices
            orientation[cell, : len(indices)] = directions
        valid = orientation != 0
        outward = (
            np.where((orientation > 0)[..., None], areas[faces], neighbor_areas[faces])
            * valid[..., None]
        )
        side_centers = np.where(
            (orientation > 0)[..., None], face_centers[faces], neighbor_centers[faces]
        )
        offsets = (side_centers - centers[:, None, :]) * valid[..., None]
        distance = contract("cfi,cfi->cf", offsets, outward) / measures[faces]
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
        area_scale = np.sum(measures[faces] * valid, axis=1)
        if np.any(np.abs(closure) > tolerance * area_scale[:, None]):
            raise ValueError("Hybrid cell outward area vectors must close.")
        defect = np.eye(width)[None] * valid[:, :, None] - contract(
            "cfi,cgi->cfg", offsets, weights
        )
        # Connectivity components are also the nullspace components of the energy.
        labels = np.arange(count)
        for left, right in zip(owner, neighbor, strict=True):
            if right >= 0:
                labels[labels == labels[right]] = labels[left]
        _, labels = np.unique(labels, return_inverse=True)
        source_faces = np.flatnonzero(active).astype(np.int32)
        quotient_rows = np.zeros(owner.size, dtype=np.int32)
        quotient_rows[source_faces] = np.arange(source_faces.size, dtype=np.int32)
        self.source_face_indices = jnp.asarray(source_faces)
        self.owner_cells = jnp.asarray(owner[source_faces])
        self.neighbor_cells = jnp.asarray(neighbor[source_faces])
        self.face_measures = jnp.asarray(measures[source_faces])
        self.face_centers = jnp.asarray(face_centers[source_faces])
        self.discretization = discretization
        self.cell_faces = jnp.asarray(quotient_rows[faces])
        self.valid = jnp.asarray(valid)
        local_slots = np.arange(faces.size, dtype=np.int32)
        self.facet_accumulation = SparseLinearMap(
            EdgeRelation(
                local_slots,
                quotient_rows[faces].reshape(-1),
                source_size=faces.size,
                target_size=source_faces.size,
                valid=valid.reshape(-1),
            ),
            jnp.ones((faces.size,), dtype=discretization.cell_volumes.dtype),
            operator_id=canonical_fingerprint(
                {
                    "kind": "hybrid-local-facet-accumulation",
                    "source": discretization.prepared_id,
                }
            ),
        )
        self.outward_areas = jnp.asarray(outward)
        self.displacements = jnp.asarray(offsets)
        self.gradient_weights = jnp.asarray(weights)
        self.defect = jnp.asarray(defect)
        self.stabilization_weights = jnp.asarray(
            measures[faces] / np.where(valid, distance, 1.0) * valid
        )
        self.owner_slots = jnp.asarray(slots[source_faces])
        self.component_ids = jnp.asarray(labels)
        self.component_count = int(labels.max()) + 1
        self.stabilization = float(stabilization)

    @property
    def cell_count(self) -> int:
        return self.discretization.cell_volumes.size

    @property
    def face_count(self) -> int:
        return self.source_face_indices.size

    def local_matrices(self, tensor: ArrayLike) -> Array:
        tensor = _positive_tensor(_tensor(tensor, self.cell_count))
        normal = self.outward_areas / self.face_measures[self.cell_faces, None]
        normal_conductivity = contract("cfi,cij,cfj->cf", normal, tensor, normal)
        penalty = self.stabilization * self.stabilization_weights * normal_conductivity
        consistent = self.discretization.cell_volumes[:, None, None] * contract(
            "cfi,cij,cgj->cfg", self.gradient_weights, tensor, self.gradient_weights
        )
        return consistent + contract("cfg,cf,cfh->cgh", self.defect, penalty, self.defect)

    def gradient(self, cell_values: ArrayLike, face_values: ArrayLike) -> Array:
        difference = (
            jnp.asarray(face_values)[self.cell_faces] - jnp.asarray(cell_values)[:, None]
        )
        return contract("cfi,cf->ci", self.gradient_weights, difference)

    def local_fluxes(
        self,
        cell_values: ArrayLike,
        face_values: ArrayLike,
        tensor: ArrayLike,
        *,
        body_force: ArrayLike | None = None,
    ) -> Array:
        tensor = _tensor(tensor, self.cell_count)
        difference = (
            jnp.asarray(face_values)[self.cell_faces] - jnp.asarray(cell_values)[:, None]
        )
        result = -contract("cfg,cg->cf", self.local_matrices(tensor), difference)
        if body_force is not None:
            force = jnp.broadcast_to(jnp.asarray(body_force), (self.cell_count, 3))
            result = result + contract(
                "cfi,cij,cj->cf", self.outward_areas, tensor, force
            )
        return jnp.where(self.valid, result, 0.0)

    def owner_rates(self, local_fluxes: Array) -> Array:
        return local_fluxes[self.owner_cells, self.owner_slots]

    def face_fluxes(
        self,
        cell_values: ArrayLike,
        face_values: ArrayLike,
        tensor: ArrayLike,
        *,
        body_force: ArrayLike | None = None,
    ) -> Array:
        return self.owner_rates(
            self.local_fluxes(cell_values, face_values, tensor, body_force=body_force)
        )

    def continuity_residual(self, local_fluxes: Array) -> Array:
        return self.facet_accumulation.mv(local_fluxes.reshape(-1))

    def cell_divergence(self, face_rates: ArrayLike) -> Array:
        """Net outward integrated owner rates, without division by cell volume."""
        owner = self.owner_cells
        neighbor = self.neighbor_cells
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
            .at[self.component_ids[self.owner_cells]]
            .max(anchored.astype(jnp.int32))
            > 0
        )

    def residual(
        self,
        cell_values: ArrayLike,
        face_values: Array,
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
            boundary.impose_dirichlet(face_values),
            tensor,
            body_force=body_force,
        )
        cells = jnp.sum(local, axis=1) - jnp.broadcast_to(
            jnp.asarray(source), (self.cell_count,)
        )
        faces = boundary.face_residual(face_values, self.continuity_residual(local))
        return jnp.concatenate((cells, faces))

    def linear_system(
        self,
        tensor: ArrayLike,
        boundary: HybridDiffusionBoundary,
        *,
        source: ArrayLike = 0.0,
        body_force: ArrayLike | None = None,
    ) -> tuple[LinearSystem, Array]:
        """Build the symmetric lifted global hybrid system; shared faces stay global."""
        size = self.cell_count + self.face_count
        zero = jnp.zeros(size, dtype=self.discretization.cell_volumes.dtype)
        zero = eqx.error_if(
            zero,
            ~jnp.all(self.anchored_components(boundary))
            | jnp.any(
                (boundary.kind == 3)
                & (~jnp.isfinite(boundary.conductance) | (boundary.conductance <= 0))
            ),
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
            # The checked local energy is SPD in valid face-cell differences:
            # SPD tensor controls gradients; positive defect penalties control
            # their complement. Shared faces identify component constants, and
            # the checked Dirichlet/positive-Robin anchors remove every one.
            properties=OperatorProperties(
                self_adjoint=True,
                positive_definite=True,
                evidence={
                    "self_adjoint": "construction",
                    "positive_definite": "construction",
                },
            ),
        )
        return LinearSystem(operator), -offset

    def pinned_linear_system(
        self,
        tensor: ArrayLike,
        boundary: HybridDiffusionBoundary,
        *,
        source: ArrayLike = 0.0,
        body_force: ArrayLike | None = None,
        pin_cell: int = 0,
        pin_value: ArrayLike = 0.0,
    ) -> tuple[LinearSystem, Array]:
        """Explicit constant gauge for a connected, compatible unanchored problem.

        ``source`` retains the existing integrated cell-load units. No mean is
        removed from a supplied source. The original physical equations remain
        valid at the pinned cell by the checked total Neumann compatibility.
        Local coefficient matrices are bound once; native diagonal actions
        eliminate the pin row/column symmetrically, preserving raw transposes.
        """
        if self.component_count != 1:
            raise ValueError(
                "A single cell pin requires one connected diffusion component."
            )
        if (
            isinstance(pin_cell, bool)
            or not isinstance(pin_cell, (int, np.integer))
            or not 0 <= pin_cell < self.cell_count
        ):
            raise ValueError("The explicit gauge pin must name an actual cell.")
        if boundary.geometry_id != self.discretization.geometry_id:
            raise ValueError("Boundary and diffusion geometry must match.")
        tensor = _positive_tensor(_tensor(tensor, self.cell_count))
        matrices = self.local_matrices(tensor)
        dtype = matrices.dtype
        loads = jnp.broadcast_to(jnp.asarray(source, dtype=dtype), (self.cell_count,))
        lift_value = jnp.asarray(pin_value, dtype=dtype)
        if lift_value.shape != ():
            raise ValueError("The cell gauge value must be scalar.")
        lift_value = eqx.error_if(
            lift_value,
            ~jnp.isfinite(lift_value)
            | jnp.any(~jnp.isfinite(loads))
            | jnp.any((boundary.kind == 1) | (boundary.kind == 3)),
            "A periodic cell gauge requires finite data and unanchored Neumann laws.",
        )
        body_rates = jnp.zeros_like(self.valid, dtype=dtype)
        if body_force is not None:
            force = jnp.broadcast_to(
                jnp.asarray(body_force, dtype=dtype), (self.cell_count, 3)
            )
            body_rates = contract("cfi,cij,cj->cf", self.outward_areas, tensor, force)

        def residual(value: Array) -> Array:
            cells, faces = value[: self.cell_count], value[self.cell_count :]
            difference = faces[self.cell_faces] - cells[:, None]
            local = jnp.where(
                self.valid, -contract("cfg,cg->cf", matrices, difference) + body_rates, 0
            )
            return jnp.concatenate(
                (
                    jnp.sum(local, axis=1) - loads,
                    boundary.face_residual(faces, self.continuity_residual(local)),
                )
            )

        size = self.cell_count + self.face_count
        zero = jnp.zeros((size,), dtype=dtype)
        offset = residual(zero)
        scale = jnp.maximum(
            1.0,
            jnp.sum(jnp.abs(loads))
            + jnp.sum(jnp.where(boundary.kind == 2, jnp.abs(boundary.value), 0)),
        )
        offset = eqx.error_if(
            offset,
            jnp.any(~jnp.isfinite(offset))
            | (jnp.abs(jnp.sum(offset)) > 256 * jnp.finfo(dtype).eps * scale),
            "Periodic diffusion source and outward Neumann flux are incompatible.",
        )
        space = ArraySpace((size,), dtype=dtype)
        homogeneous_boundary = eqx.tree_at(
            lambda value: value.value, boundary, jnp.zeros_like(boundary.value)
        )

        def physical_action(value: Array) -> Array:
            difference = (
                value[self.cell_count :][self.cell_faces] - value[: self.cell_count, None]
            )
            local = jnp.where(
                self.valid, -contract("cfg,cg->cf", matrices, difference), 0
            )
            return jnp.concatenate(
                (
                    jnp.sum(local, axis=1),
                    homogeneous_boundary.face_residual(
                        value[self.cell_count :], self.continuity_residual(local)
                    ),
                )
            )

        physical = FunctionLinearOperator(
            physical_action,
            source=space,
            target=space,
            properties=OperatorProperties(self_adjoint=True, positive_semidefinite=True),
            operator_id=canonical_fingerprint(
                {
                    "kind": "hybrid-physical-diffusion",
                    "source": self.discretization.prepared_id,
                }
            ),
        )
        free_diagonal = jnp.ones((size,), dtype=dtype).at[pin_cell].set(0)
        free = DiagonalLinearOperator(
            free_diagonal,
            space=space,
            operator_id=canonical_fingerprint(
                {
                    "kind": "hybrid-free-cell-gauge",
                    "source": self.discretization.prepared_id,
                    "pin_cell": int(pin_cell),
                }
            ),
        )
        pinned = DiagonalLinearOperator(
            1 - free_diagonal,
            space=space,
            operator_id=canonical_fingerprint(
                {
                    "kind": "hybrid-pinned-cell-gauge",
                    "source": self.discretization.prepared_id,
                    "pin_cell": int(pin_cell),
                }
            ),
        )
        operator = SumLinearOperator(
            ComposedLinearOperator(free, ComposedLinearOperator(physical, free)),
            pinned,
        )
        operator = eqx.tree_at(
            lambda value: value.properties,
            operator,
            OperatorProperties(
                self_adjoint=True,
                positive_definite=True,
                evidence={
                    "self_adjoint": "construction",
                    "positive_definite": "construction",
                },
            ),
        )
        lift = zero.at[pin_cell].set(lift_value)
        rhs = free.mv(-offset - physical.mv(lift)) + lift
        return LinearSystem(operator), rhs

    def trace_reconstruction_operators(
        self,
        tensor: ArrayLike,
        boundary: HybridDiffusionBoundary,
        *,
        body_force: ArrayLike | None = None,
    ) -> tuple[LinearSystem, FunctionLinearOperator, Array]:
        """Bind one face stationarity system and its affine cell-state RHS.

        The returned RHS action maps actual cell potentials to actual facet
        loads. Factor the face system once; rebind each transferred cell/history
        state through that action plus the returned physical-load offset.
        This is equation-owned trace reprepare, not interpolation or conservative
        remapping of old facet coefficients.
        """
        if (
            boundary.geometry_id != self.discretization.geometry_id
            or boundary.kind.shape != (self.face_count,)
        ):
            raise ValueError(
                "Trace reprepare requires boundary laws on the actual quotient facet axis."
            )
        tensor = _tensor(tensor, self.cell_count)
        matrices = self.local_matrices(tensor)
        dtype = matrices.dtype
        body_rates = jnp.zeros_like(self.valid, dtype=dtype)
        if body_force is not None:
            force = jnp.broadcast_to(
                jnp.asarray(body_force, dtype=dtype), (self.cell_count, 3)
            )
            body_rates = contract("cfi,cij,cj->cf", self.outward_areas, tensor, force)
        zero = jnp.zeros((self.face_count,), dtype=dtype)
        boundary_difference = boundary.impose_dirichlet(zero)[self.cell_faces]
        boundary_local = jnp.where(
            self.valid,
            -contract("cfg,cg->cf", matrices, boundary_difference) + body_rates,
            0,
        )
        offset = boundary.face_residual(zero, self.continuity_residual(boundary_local))
        offset = eqx.error_if(
            offset,
            jnp.any(~jnp.isfinite(offset)),
            "Trace reprepare requires finite physical boundary and material data.",
        )
        space = ArraySpace((self.face_count,), dtype=dtype)
        homogeneous_boundary = eqx.tree_at(
            lambda value: value.value, boundary, jnp.zeros_like(boundary.value)
        )

        def face_action(faces: Array) -> Array:
            difference = homogeneous_boundary.impose_dirichlet(faces)[self.cell_faces]
            local = jnp.where(
                self.valid, -contract("cfg,cg->cf", matrices, difference), 0
            )
            return homogeneous_boundary.face_residual(
                faces, self.continuity_residual(local)
            )

        def cell_rhs(cells: Array) -> Array:
            cells = eqx.error_if(
                cells,
                jnp.any(~jnp.isfinite(cells)),
                "Transferred cell potentials must be finite.",
            )
            difference = jnp.broadcast_to(cells[:, None], self.valid.shape)
            local = jnp.where(self.valid, contract("cfg,cg->cf", matrices, difference), 0)
            return jnp.where(boundary.kind == 1, 0, self.continuity_residual(local))

        operator = FunctionLinearOperator(
            face_action,
            source=space,
            target=space,
            properties=OperatorProperties(
                self_adjoint=True,
                positive_definite=True,
                evidence={
                    "self_adjoint": "construction",
                    "positive_definite": "construction",
                },
            ),
            operator_id=canonical_fingerprint(
                {
                    "kind": "hybrid-facet-stationarity",
                    "source": self.discretization.prepared_id,
                }
            ),
        )
        rhs_action = FunctionLinearOperator(
            cell_rhs,
            source=ArraySpace((self.cell_count,), dtype=dtype),
            target=space,
            transpose_action=eqx.Partial(
                _trace_cell_rhs_transpose,
                matrices,
                self.facet_accumulation,
                boundary.kind,
                self.valid.shape,
            ),
            operator_id=canonical_fingerprint(
                {
                    "kind": "hybrid-transferred-cell-trace-load",
                    "source": self.discretization.prepared_id,
                }
            ),
        )
        return LinearSystem(operator), rhs_action, -offset

    def trace_linear_system(
        self,
        cell_values: ArrayLike,
        tensor: ArrayLike,
        boundary: HybridDiffusionBoundary,
        *,
        body_force: ArrayLike | None = None,
    ) -> tuple[LinearSystem, Array]:
        """Reprepare physical traces for one supplied cell-state right hand side."""
        system, rhs_action, offset = self.trace_reconstruction_operators(
            tensor,
            boundary,
            body_force=body_force,
        )
        cells = jnp.asarray(cell_values, dtype=offset.dtype)
        if cells.shape != (self.cell_count,):
            raise ValueError(
                "Trace reprepare requires one scalar potential per actual cell."
            )
        return system, rhs_action.mv(cells) + offset

    def solve(
        self,
        tensor: ArrayLike,
        boundary: HybridDiffusionBoundary,
        *,
        source: ArrayLike = 0.0,
        body_force: ArrayLike | None = None,
        policy: LinearSolvePolicy | None = None,
    ) -> LinearSolveResult:
        system, rhs = self.linear_system(
            tensor, boundary, source=source, body_force=body_force
        )
        return solve(
            system,
            rhs,
            policy=LinearSolvePolicy(ConjugateGradient()) if policy is None else policy,
        )


__all__ = ["HybridMimeticDiffusion"]
