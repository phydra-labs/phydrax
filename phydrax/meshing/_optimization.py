#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._physical import SpatialCoordinateContract
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellGeometrySpec, CellMesh
from ._canonical import certify_cell_mesh
from ._quality import evaluate_cell_quality
from ._result import CellMeshingResult


class TargetMatrixOptimizationPlan(StrictModule, NonTrainableState):
    mesh: CellMesh
    target_coordinates: Array
    target_inverses: tuple[Array | None, ...]
    fixed_vertices: Array
    maximum_iterations: int = eqx.field(static=True)
    initial_step_size: float = eqx.field(static=True)
    minimum_mean_ratio: float = eqx.field(static=True)
    movement_weight: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh: CellMesh,
        /,
        *,
        target_coordinates: ArrayLike | None = None,
        fixed_vertices: ArrayLike | None = None,
        maximum_iterations: int = 50,
        initial_step_size: float = 0.05,
        minimum_mean_ratio: float = 1.0e-6,
        movement_weight: float = 1.0e-4,
    ):
        if not isinstance(mesh, CellMesh):
            raise TypeError("mesh must be CellMesh.")
        target = np.asarray(
            mesh.coordinates if target_coordinates is None else target_coordinates,
            dtype=float,
        )
        fixed = (
            np.zeros((mesh.coordinates.shape[0],), dtype=bool)
            if fixed_vertices is None
            else np.asarray(fixed_vertices, dtype=bool)
        )
        iterations = int(maximum_iterations)
        step = float(initial_step_size)
        floor = float(minimum_mean_ratio)
        movement = float(movement_weight)
        if target.shape != mesh.coordinates.shape or not np.all(np.isfinite(target)):
            raise ValueError("target_coordinates must match the finite mesh coordinates.")
        if fixed.shape != (mesh.coordinates.shape[0],):
            raise ValueError("fixed_vertices must match the mesh vertex count.")
        if iterations <= 0 or not np.isfinite(step) or step <= 0.0:
            raise ValueError("Optimization iterations and step size must be positive.")
        if not np.isfinite(floor) or floor <= 0.0 or floor >= 1.0:
            raise ValueError("minimum_mean_ratio must lie strictly between zero and one.")
        if not np.isfinite(movement) or movement < 0.0:
            raise ValueError("movement_weight must be finite and non-negative.")
        inverses = []
        for block in mesh.blocks:
            dimension = block.topological_dimension
            if (
                block.cell_kind not in ("triangle", "tetrahedron")
                or mesh.ambient_dimension != dimension
            ):
                inverses.append(None)
                continue
            values = target[np.asarray(block.vertices, dtype=np.int32)]
            matrix = np.stack(
                tuple(
                    values[:, index] - values[:, 0] for index in range(1, dimension + 1)
                ),
                axis=-1,
            )
            determinant = np.linalg.det(matrix)
            if np.any(~np.isfinite(determinant)) or np.any(determinant <= 0.0):
                raise ValueError("Target simplex matrices must be positively oriented.")
            inverses.append(jnp.asarray(np.linalg.inv(matrix)))
        self.mesh = mesh
        self.target_coordinates = jnp.asarray(target)
        self.target_inverses = tuple(inverses)
        self.fixed_vertices = jnp.asarray(fixed)
        self.maximum_iterations = iterations
        self.initial_step_size = step
        self.minimum_mean_ratio = floor
        self.movement_weight = movement
        self.plan_id = canonical_fingerprint(
            {
                "kind": "target-matrix-optimization-plan",
                "mesh": mesh.mesh_id,
                "target_coordinates": array_tree_fingerprint(target),
                "fixed_vertices": array_tree_fingerprint(fixed),
                "maximum_iterations": iterations,
                "initial_step_size": step,
                "minimum_mean_ratio": floor,
                "movement_weight": movement,
            }
        )

    def objective(self, coordinates: ArrayLike, /) -> Array:
        points = jnp.asarray(coordinates)
        quality = evaluate_cell_quality(self.mesh, points)
        safe_ratio = jnp.maximum(
            quality.mean_ratios - self.minimum_mean_ratio,
            jnp.finfo(points.dtype).tiny,
        )
        barrier = -jnp.sum(jnp.log(safe_ratio))
        invalid_penalty = jnp.sum(
            jnp.where(quality.valid, 0.0, jnp.asarray(1.0e12, dtype=points.dtype))
        )
        target_energy = jnp.asarray(0.0, dtype=points.dtype)
        for block, inverse in zip(self.mesh.blocks, self.target_inverses, strict=True):
            values = points[jnp.asarray(block.vertices, dtype=jnp.int32)]
            if inverse is None:
                target_values = self.target_coordinates[
                    jnp.asarray(block.vertices, dtype=jnp.int32)
                ]
                target_energy = target_energy + jnp.sum((values - target_values) ** 2)
                continue
            dimension = block.topological_dimension
            matrix = jnp.stack(
                tuple(
                    values[:, index] - values[:, 0] for index in range(1, dimension + 1)
                ),
                axis=-1,
            )
            transform = ein.contract("cai,cij->caj", matrix, inverse, backend="jax")
            gram = ein.contract("cai,caj->cij", transform, transform, backend="jax")
            identity = jnp.eye(dimension, dtype=points.dtype)
            shape = jnp.sum((gram - identity) ** 2)
            size = jnp.sum((jnp.linalg.det(transform) - 1.0) ** 2)
            target_energy = target_energy + shape + size
        movement = self.movement_weight * jnp.sum((points - self.target_coordinates) ** 2)
        return target_energy + barrier + movement + invalid_penalty


class MeshOptimizationResult(StrictModule, NonTrainableState):
    result: CellMeshingResult
    initial_objective: float = eqx.field(static=True)
    final_objective: float = eqx.field(static=True)
    iterations: int = eqx.field(static=True)
    accepted_steps: int = eqx.field(static=True)
    optimization_id: str = eqx.field(static=True)


def optimize_cell_geometry_coordinates(
    geometry: CellGeometrySpec,
    objective: Callable[[Array], Array],
    /,
    *,
    fixed_coordinates: ArrayLike | None = None,
    maximum_iterations: int = 50,
    initial_step_size: float = 0.05,
) -> Array:
    """Optimize arbitrary high-order geometry coordinates under fixed topology."""

    if not isinstance(geometry, CellGeometrySpec):
        raise TypeError("geometry must be CellGeometrySpec.")
    if not callable(objective):
        raise TypeError("objective must be callable.")
    coordinates = jnp.asarray(geometry.coordinates)
    fixed = (
        jnp.zeros((coordinates.shape[0],), dtype=bool)
        if fixed_coordinates is None
        else jnp.asarray(fixed_coordinates, dtype=bool)
    )
    if fixed.shape != (coordinates.shape[0],):
        raise ValueError("fixed_coordinates must match geometry coordinate count.")
    step = float(initial_step_size)
    for _ in range(int(maximum_iterations)):
        value, gradient = jax.value_and_grad(objective)(coordinates)
        gradient = jnp.where(fixed[:, None], 0.0, gradient)
        candidate = jnp.where(
            fixed[:, None],
            coordinates,
            coordinates - step * gradient,
        )
        candidate_value = objective(candidate)
        accepted = jnp.isfinite(candidate_value) & (candidate_value < value)
        coordinates = jnp.where(accepted, candidate, coordinates)
        step = float(step * (1.05 if bool(accepted) else 0.5))
    return coordinates


def optimize_cell_mesh(
    plan: TargetMatrixOptimizationPlan,
    coordinate_contract: SpatialCoordinateContract,
    /,
    *,
    project: Callable[[Array], Array] | None = None,
    numeric_version: str = "mesh-optimized",
) -> MeshOptimizationResult:
    """Run safeguarded fixed-topology optimization and certify the result."""

    if not isinstance(plan, TargetMatrixOptimizationPlan):
        raise TypeError("plan must be TargetMatrixOptimizationPlan.")
    if not isinstance(coordinate_contract, SpatialCoordinateContract):
        raise TypeError("coordinate_contract must be SpatialCoordinateContract.")
    if project is not None and not callable(project):
        raise TypeError("project must be callable or None.")
    coordinates = jnp.asarray(plan.mesh.coordinates)
    initial = float(plan.objective(coordinates))
    step = plan.initial_step_size
    accepted_steps = 0
    iterations = 0
    for iteration in range(plan.maximum_iterations):
        value, gradient = jax.value_and_grad(plan.objective)(coordinates)
        gradient = jnp.where(plan.fixed_vertices[:, None], 0.0, gradient)
        candidate = jnp.where(
            plan.fixed_vertices[:, None],
            coordinates,
            coordinates - step * gradient,
        )
        if project is not None:
            projected = jnp.asarray(project(candidate))
            if projected.shape != candidate.shape:
                raise ValueError("Mesh projection must preserve coordinate shape.")
            candidate = jnp.where(plan.fixed_vertices[:, None], coordinates, projected)
        candidate_quality = evaluate_cell_quality(plan.mesh, candidate)
        candidate_value = plan.objective(candidate)
        accepted = bool(
            jnp.isfinite(candidate_value)
            & jnp.all(candidate_quality.valid)
            & (candidate_value < value)
        )
        if accepted:
            coordinates = candidate
            accepted_steps += 1
            step *= 1.05
        else:
            step *= 0.5
        iterations = iteration + 1
        if step < 1.0e-12:
            break
    final = float(plan.objective(coordinates))
    optimized = plan.mesh.with_coordinates(
        coordinates,
        numeric_version=numeric_version,
    )
    result = certify_cell_mesh(optimized, coordinate_contract)
    optimization_id = canonical_fingerprint(
        {
            "kind": "mesh-optimization-result",
            "plan": plan.plan_id,
            "result": result.result_id,
            "initial_objective": initial,
            "final_objective": final,
            "iterations": iterations,
            "accepted_steps": accepted_steps,
        }
    )
    return MeshOptimizationResult(
        result=result,
        initial_objective=initial,
        final_objective=final,
        iterations=iterations,
        accepted_steps=accepted_steps,
        optimization_id=optimization_id,
    )


__all__ = [
    "MeshOptimizationResult",
    "TargetMatrixOptimizationPlan",
    "optimize_cell_geometry_coordinates",
    "optimize_cell_mesh",
]
