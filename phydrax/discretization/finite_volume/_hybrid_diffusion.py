#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Affine-consistent, coercive hybrid mimetic diffusion on admissible 3-D FV cells."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._strict import StrictModule
from ...ein import contract
from ...linalg import (
    ArraySpace,
    ConjugateGradient,
    FunctionLinearOperator,
    LinearSolvePolicy,
    LinearSystem,
    OperatorProperties,
    solve,
)
from ._diffusion_boundary import HybridDiffusionBoundary
from ._unstructured import UnstructuredFiniteVolumeDiscretization


def _tensor(value, count: int) -> Array:
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


class HybridMimeticDiffusion(StrictModule):
    """Fixed polyhedral geometry, shared global face potentials, no TPFA reduction.

    Cells must be strictly star-shaped about their FV centers, have planar faces,
    positive measures, outward closure and exact first geometric moments. The
    local energy is V grad(u).K.grad(u) plus a positive normal-distance weighted
    penalty on deviations from affine face traces. This is consistent for full
    rotated SPD tensors, including off-diagonal tensor entries.
    """

    discretization: UnstructuredFiniteVolumeDiscretization
    cell_faces: Array
    valid: Array
    outward_areas: Array
    displacements: Array
    gradient_weights: Array
    defect: Array
    stabilization_weights: Array
    owner_slots: Array
    component_ids: Array
    component_count: int = eqx.field(static=True)
    stabilization: float = eqx.field(static=True)

    def __init__(self, discretization, /, *, stabilization: float = 1.0):
        if not isinstance(discretization, UnstructuredFiniteVolumeDiscretization):
            raise TypeError("Hybrid diffusion requires native prepared unstructured FV.")
        if discretization.cell_dimension != 3:
            raise ValueError("Hybrid mimetic diffusion requires three-dimensional cells.")
        if not np.isfinite(stabilization) or stabilization <= 0:
            raise ValueError("stabilization must be finite and strictly positive.")
        owner = np.asarray(discretization.owner_cells)
        neighbour = np.asarray(discretization.neighbour_cells)
        centers = np.asarray(discretization.cell_centers)
        volumes = np.asarray(discretization.cell_volumes)
        face_centers = np.asarray(discretization.face_centers)
        areas = np.asarray(discretization.area_vectors)
        measures = np.asarray(discretization.face_measures)
        count = volumes.size
        lists = [[] for _ in range(count)]
        signs = [[] for _ in range(count)]
        slots = np.zeros(owner.size, dtype=np.int32)
        for face, (left, right) in enumerate(zip(owner, neighbour, strict=True)):
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
        outward = areas[faces] * orientation[..., None]
        offsets = (face_centers[faces] - centers[:, None, :]) * valid[..., None]
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
        for left, right in zip(owner, neighbour, strict=True):
            if right >= 0:
                labels[labels == labels[right]] = labels[left]
        _, labels = np.unique(labels, return_inverse=True)
        self.discretization = discretization
        self.cell_faces = jnp.asarray(faces)
        self.valid = jnp.asarray(valid)
        self.outward_areas = jnp.asarray(outward)
        self.displacements = jnp.asarray(offsets)
        self.gradient_weights = jnp.asarray(weights)
        self.defect = jnp.asarray(defect)
        self.stabilization_weights = jnp.asarray(
            measures[faces] / np.where(valid, distance, 1.0) * valid
        )
        self.owner_slots = jnp.asarray(slots)
        self.component_ids = jnp.asarray(labels)
        self.component_count = int(labels.max()) + 1
        self.stabilization = float(stabilization)

    @property
    def cell_count(self):
        return self.discretization.cell_volumes.size

    @property
    def face_count(self):
        return self.discretization.face_measures.size

    def local_matrices(self, tensor) -> Array:
        tensor = _positive_tensor(_tensor(tensor, self.cell_count))
        normal = (
            self.outward_areas / self.discretization.face_measures[self.cell_faces, None]
        )
        normal_conductivity = contract("cfi,cij,cfj->cf", normal, tensor, normal)
        penalty = self.stabilization * self.stabilization_weights * normal_conductivity
        consistent = self.discretization.cell_volumes[:, None, None] * contract(
            "cfi,cij,cgj->cfg", self.gradient_weights, tensor, self.gradient_weights
        )
        return consistent + contract("cfg,cf,cfh->cgh", self.defect, penalty, self.defect)

    def gradient(self, cell_values, face_values) -> Array:
        difference = (
            jnp.asarray(face_values)[self.cell_faces] - jnp.asarray(cell_values)[:, None]
        )
        return contract("cfi,cf->ci", self.gradient_weights, difference)

    def local_fluxes(self, cell_values, face_values, tensor, *, body_force=None) -> Array:
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

    def owner_rates(self, local_fluxes) -> Array:
        return local_fluxes[self.discretization.owner_cells, self.owner_slots]

    def face_fluxes(self, cell_values, face_values, tensor, *, body_force=None) -> Array:
        return self.owner_rates(
            self.local_fluxes(cell_values, face_values, tensor, body_force=body_force)
        )

    def continuity_residual(self, local_fluxes) -> Array:
        return (
            jnp.zeros(self.face_count, dtype=local_fluxes.dtype)
            .at[self.cell_faces]
            .add(jnp.where(self.valid, local_fluxes, 0.0))
        )

    def cell_divergence(self, face_rates) -> Array:
        """Net outward integrated owner rates, without division by cell volume."""
        owner = self.discretization.owner_cells
        neighbour = self.discretization.neighbour_cells
        rates = jnp.asarray(face_rates)
        result = jnp.zeros(self.cell_count, dtype=rates.dtype).at[owner].add(rates)
        return result.at[jnp.maximum(neighbour, 0)].add(
            jnp.where(neighbour >= 0, -rates, 0.0)
        )

    def anchored_components(self, boundary) -> Array:
        anchored = (boundary.kind == 1) | (
            (boundary.kind == 3) & (boundary.conductance > 0)
        )
        return (
            jnp.zeros(self.component_count, dtype=jnp.int32)
            .at[self.component_ids[self.discretization.owner_cells]]
            .max(anchored.astype(jnp.int32))
            > 0
        )

    def residual(
        self, cell_values, face_values, tensor, boundary, *, source=0.0, body_force=None
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

    def linear_system(self, tensor, boundary, *, source=0.0, body_force=None):
        """Build the symmetric lifted global hybrid system; shared faces stay global."""
        size = self.cell_count + self.face_count
        zero = jnp.zeros(size, dtype=self.discretization.cell_volumes.dtype)
        zero = eqx.error_if(
            zero,
            ~jnp.all(self.anchored_components(boundary)),
            "Every diffusion component needs Dirichlet or positive Robin anchoring.",
        )

        def residual(value):
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
            properties=OperatorProperties(self_adjoint=True, positive_definite=True),
        )
        return LinearSystem(operator), -offset

    def solve(
        self,
        tensor,
        boundary: HybridDiffusionBoundary,
        *,
        source=0.0,
        body_force=None,
        policy=None,
    ):
        system, rhs = self.linear_system(
            tensor, boundary, source=source, body_force=body_force
        )
        return solve(
            system,
            rhs,
            policy=LinearSolvePolicy(ConjugateGradient()) if policy is None else policy,
        )


__all__ = ["HybridMimeticDiffusion"]
